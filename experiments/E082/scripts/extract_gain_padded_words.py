"""Archive gain-corrected short-payload observations for parity experiments."""

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_gain_transfer import decisions, four_corner_slots
from audit_pn_padding_starts import CacheOnly, lattice_model, prefix_profile
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


PREFIX = 1024


def conditioned_prefix(received, visible, reference, conservative_bound, prefix):
    bound = (conservative_bound + 7) // 8 * 8
    if bound > prefix:
        raise ValueError("Padding begins after the analysis prefix")
    bits, mask = received[:prefix].copy(), visible[:prefix].copy()
    bits[bound:] = reference[bound:prefix]
    mask[bound:] = True
    return bits, mask, bound


def constellation_models(cross, inputs):
    with np.load(checked_artifact("v026-ut-qam-extract-repeat", "E028", "ut_qam_symbols.npz", inputs),
                 allow_pickle=False) as data:
        p16, m16 = data["constellation_points"], data["frozen_gray_mapping"]
    amplitude = np.ptp(m16[np.isclose(abs(p16), abs(p16).max())], axis=0) == 0
    points32 = np.array([complex(p["i"], p["q"]) for p in cross["cross32_mapping"]])
    labels32 = labels_as_bits(np.array([p["label"] for p in cross["cross32_mapping"]]), 5)
    return {"16QAM": (lattice_model(p16, m16), amplitude),
            "cross32": (lattice_model(points32, labels32), np.zeros(5, dtype=bool))}


def gather_words(archive, gain, layout, models, pn):
    source = archive["yDataDec"]
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    selected = defaultdict(list)
    data = defaultdict(list)
    frames = layout["word_windows_and_onsets"]

    @lru_cache(maxsize=1)
    def block(first):
        last = min(first + source.chunks[0], len(source))
        return as_complex(source[first:last, 1:, :])[..., NONPILOT % 1024]

    for row, geom in zip(gain["frames"], frames, strict=True):
        index = row["frame"]
        if geom["frame"] != index:
            raise ValueError("Archive frame orders differ")
        requested = []
        for obs, alloc in zip(row["allocations"], geom["allocations"], strict=True):
            if obs["start"] != alloc["start"] or obs["label"] != alloc["label"]:
                raise ValueError("Archive allocation coordinates differ")
            count = alloc["count"]
            if count - 1 in obs["masked_transfer_stop_words"]:
                requested.append(("short", alloc, count - 1))
            if obs["label"] in (66, 70) and count >= 2:
                requested.append(("nonterminal", alloc, 0))
        if not requested:
            continue
        first = index // source.chunks[0] * source.chunks[0]
        flat = remove_header_gain(block(first)[index - first] * rotation, signs)
        axis = complex(*row["axis"])
        corner = four_corner_slots(flat, axis)
        for role, alloc, word in requested:
            label = alloc["label"]
            start = alloc["start"] + word * alloc["width_resources"]
            stop = layout["marker_systematic_ends"][str(label)]
            bits, seen, _, _ = decisions(flat, axis, {**alloc, "start": start, "count": 1}, models, stop, corner)
            bits, seen = bits[0], seen[0]
            record = {"frame": index, "group": row["group"], "calibration": row["calibration"],
                      "resource": start, "word": word, "edge": stop, "length": len(bits)}
            key = f"{role}_{label}"
            if role == "short":
                result = prefix_profile(bits, seen, pn, stop)
                onset = result["two_nat_minmax"][1]
                if onset > PREFIX:
                    continue
                conditioned, conditional_mask, bound = conditioned_prefix(bits, seen, pn, onset, PREFIX)
                mask = seen[bound:stop]
                errors = (bits[bound:stop] != pn[bound:stop]) & mask
                record.update(bound=bound, pn_seen=int(mask.sum()), pn_errors=int(errors.sum()))
                data[key + "_conditioned"].append(np.packbits(conditioned))
                data[key + "_conditional_mask"].append(np.packbits(conditional_mask))
            selected[key].append(record)
            data[key + "_bits"].append(np.packbits(bits & seen))
            data[key + "_seen"].append(np.packbits(seen))
        if index % 130 == 0:
            print("Read frame", index, flush=True)
    return dict(selected), {key: np.array(values) for key, values in data.items()}


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    gain = load("v079-code5-gain-masked", "E081", "code5_gain_transfer.json")
    layout = load("v068-pn-padding-starts", "E070", "pn_padding_starts.json")
    cross = load("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json")
    models = constellation_models(cross, inputs)
    pn = equality_predictions(np.arange(16385), pn_bits(), cross["frozen_pn_model"])
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        words, observations = gather_words(archive, gain, layout, models, pn)
    out = experiment_output("E082")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "gain_padded_words.npz", **observations)
    report = {"schema": "starlink-gain-padded-words/v1", "prefix_bits": PREFIX, "cohorts": words,
              "source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                         "pages": [reader.receipts[page] for page in sorted(reader.receipts)]},
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    (out / "gain_padded_words.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print({key: len(value) for key, value in words.items()})


if __name__ == "__main__":
    main()
