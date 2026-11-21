"""Collect the gain-corrected terminal MCS 66 words at all observed PN onsets."""

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
from audit_pn_padding_starts import CacheOnly, prefix_profile
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


def measure_tail(bits, seen, pn, edge):
    profile = prefix_profile(bits, seen, pn, edge)
    lower, upper = profile["two_nat_minmax"]
    bound = (upper + 7) // 8 * 8
    observed = seen[bound:edge]
    errors = (bits[bound:edge] != pn[bound:edge]) & observed
    return {"onset": profile["ml_start"], "onset_two_nat_interval": [lower, upper], "bound": bound,
            "pn_seen": int(observed.sum()), "pn_errors": int(errors.sum())}


def extract(source, gain, layout, models, pn):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    edge = layout["marker_systematic_ends"]["66"]
    rows, bits, masks = [], [], []

    @lru_cache(maxsize=1)
    def block(first):
        return as_complex(source[first:min(first + source.chunks[0], len(source)), 1:, :])[..., NONPILOT % 1024]

    for row, geometry in zip(gain["frames"], layout["word_windows_and_onsets"], strict=True):
        index = row["frame"]
        if index != geometry["frame"]:
            raise ValueError("Archive frame order differs")
        allocations = [(obs, alloc) for obs, alloc in zip(row["allocations"], geometry["allocations"], strict=True)
                       if alloc["label"] == 66 and alloc["count"] - 1 in obs["masked_transfer_stop_words"]]
        if not allocations:
            continue
        first = index // source.chunks[0] * source.chunks[0]
        flat = remove_header_gain(block(first)[index - first] * rotation, signs)
        axis = complex(*row["axis"])
        corners = four_corner_slots(flat, axis)
        for obs, alloc in allocations:
            if obs["start"] != alloc["start"] or obs["label"] != alloc["label"]:
                raise ValueError("Archive allocation coordinate differs")
            start = alloc["start"] + (alloc["count"] - 1) * alloc["width_resources"]
            observed, seen, _, _ = decisions(flat, axis, {**alloc, "start": start, "count": 1}, models, edge, corners)
            observed, seen = observed[0], seen[0]
            bits.append(np.packbits(observed & seen))
            masks.append(np.packbits(seen))
            rows.append({"frame": index, "group": row["group"], "calibration": row["calibration"], "resource": start,
                         **measure_tail(observed, seen, pn, edge)})
    return edge, rows, np.array(bits), np.array(masks)


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    gain = load("v079-code5-gain-masked", "E081", "code5_gain_transfer.json")
    geometry = load("v068-pn-padding-starts", "E070", "pn_padding_starts.json")
    cross = load("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json")
    model = cross["frozen_pn_model"]
    pn = equality_predictions(np.arange(16384), pn_bits(), model)
    models = constellation_models(cross, inputs)
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        edge, rows, bits, masks = extract(archive["yDataDec"], gain, geometry, models, pn)
    out = experiment_output("E083")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs66_padding_words.npz", bits=bits, mask=masks)
    report = {"schema": "starlink-mcs66-padding-sweep/v1", "systematic_end": edge, "codeword_bits": 16384,
              "rows": rows, "pn_model": model,
              "source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                         "pages": [reader.receipts[i] for i in sorted(reader.receipts)]},
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    (out / "mcs66_padding_words.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("MCS 66 screened terminal words", len(rows))


if __name__ == "__main__":
    main()
