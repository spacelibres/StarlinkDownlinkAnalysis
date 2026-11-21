"""Transfer frozen QC families by systematic-region alignment."""

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
from audit_cross_mcs_checks import BIT_LENGTH, LIFT, checks, paired_syndromes
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import collect
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex
from sparse_code5_core import binary_rank


SOURCES = {66: 12288, 70: 12800}
TARGETS = (66, 70, 71, 75, 78, 81, 85, 88, 92, 93)


def translate_rows(rows, source_end, target_end):
    systematic = rows < source_end
    targets = np.where(systematic, rows + target_end - source_end, rows)
    if np.any(targets < 0) or np.any((~systematic) & (targets < target_end)):
        return None
    return targets


def middle_words(data, gain, layout, models):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    prefix = np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)]
    signs = 1 - 2 * encode_block(prefix)[:48].astype(float)
    output = defaultdict(list)

    @lru_cache(maxsize=1)
    def block(first):
        stop = min(first + data.chunks[0], len(data))
        return as_complex(data[first:stop, 1:, :])[..., NONPILOT % 1024]

    for r, frame in zip(gain["frames"], layout["word_windows_and_onsets"], strict=True):
        i = r["frame"]
        if i != frame["frame"]:
            raise ValueError("Allocation frame does not match gain frame")
        candidates = [a for a in frame["allocations"] if a["label"] in TARGETS and a["count"] >= 3]
        if not candidates:
            continue
        first = i // data.chunks[0] * data.chunks[0]
        flat = remove_header_gain(block(first)[i - first] * rotation, signs)
        axis = complex(*r["axis"])
        corners = four_corner_slots(flat, axis)
        for a in candidates:
            word = a["count"] // 2
            start = a["start"] + word * a["width_resources"]
            code = a["label"]
            edge = layout["marker_systematic_ends"][str(code)]
            b, m, _, _ = decisions(flat, axis, {**a, "start": start, "count": 1}, models, edge, corners)
            bits, mask = b[0, :BIT_LENGTH], m[0, :BIT_LENGTH]
            output[code].append(({"frame": i, "resource": start, "group": r["group"], "count": a["count"],
                                  "word": word, "calibration": r["calibration"]},
                                 np.packbits(bits & mask), np.packbits(mask)))
    return output


def record(bits, masks, meta, role, excluded):
    held = np.array([not r["calibration"] and r["group"] not in excluded for r in meta])
    return role, bits[held], masks[held], np.array([r["group"] for r in meta])[held]


def original_words(artifact):
    name = "v026-ut-qam-extract-repeat"
    src = artifact(name, "E028", "ut_qam_symbols.npz")
    meta = read(artifact(name, "E028", "ut_qam_extraction.json"))
    raw, _, _ = collect(src, meta)
    output = {}
    for code in (66, 70):
        bits, _, _, rows = raw[code]
        info = [{**r, "group": r["frame"]} for r in rows]
        output[code] = record(bits.astype(bool), np.ones_like(bits, dtype=bool), info, "raw250_16qam", ())
    return output


def endpoint_words(data, meta, code, excluded):
    for source in ("archive", "raw250"):
        for role in ("nonterminal", "terminal"):
            name = f"{code}_{role}"
            if name not in meta[source]:
                continue
            rows = meta[source][name]["rows"]
            b, m = (np.unpackbits(data[f"{source}_{name}_{k}"], axis=1)[:, :BIT_LENGTH].astype(bool)
                    for k in ("bits", "seen"))
            yield record(b, m, rows, f"{source}_{role}", excluded if source == "archive" else ())


def scores(support, wrong, sets):
    out = {}
    for name, bits, seen, groups in sets:
        if len(bits) == 0:
            continue
        main, control = paired_syndromes(bits, seen, groups, support, wrong)
        out[name] = {"words": len(bits), "available_groups": len(np.unique(groups)), "reference": main,
                     "shifted_systematic": control}
    return out


def shift_systematic(support, edge):
    return np.where(support < edge, support // LIFT * LIFT + (support + 1) % LIFT, support)


def augment(old, additional):
    rows = np.vstack((old, additional))
    matrix = np.zeros((len(rows), BIT_LENGTH), dtype=np.uint8)
    matrix[np.arange(len(rows))[:, None], rows] = 1
    return rows, matrix, binary_rank(matrix), binary_rank(matrix[:, SOURCES[66]:].copy())


def main():
    require_run_context()
    inputs, saved, middle = {}, {}, {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    families = checks(art)
    with np.load(art("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz"), allow_pickle=False) as z:
        families[66] = np.sort(z["row_support"], axis=1).reshape(5, LIFT, 14)
    aligned = translate_rows(families[66][4], SOURCES[66], SOURCES[70])
    if not np.array_equal(aligned, families[70][0]):
        raise ValueError("Frozen shared row no longer gives the systematic translation")
    additional = translate_rows(families[70][1], SOURCES[70], SOURCES[66])
    support, matrix, rank, parity_rank = augment(families[66].reshape(-1, 14), additional)
    if rank != len(support):
        raise ValueError("Translated row family did not add 512 independent checks")
    cancelled = np.stack([sorted(set(a) ^ set(b)) for a, b in zip(families[66][1], families[66][3], strict=True)])
    labels = read(art("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    models = constellation_models(labels, inputs)
    gain = read(art("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    layout = read(art("v068-pn-padding-starts", "E070", "pn_padding_starts.json"))
    exclusions = read(art("v122-mcs85-archive-records-verified", "E124", "mcs85_archive_records.json"))
    excluded = exclusions["archive_overlapping_groups_excluded"]
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        measurements = middle_words(archive["yDataDec"], gain, layout, models)
    measurements_meta = {}
    for code, observed in measurements.items():
        rows, bits, masks = zip(*observed, strict=True)
        measurements_meta[code] = rows
        saved[f"mcs{code}_bits"] = b = np.stack(bits)
        saved[f"mcs{code}_seen"] = m = np.stack(masks)
        middle[code] = record(np.unpackbits(b, axis=1).astype(bool), np.unpackbits(m, axis=1).astype(bool),
                              rows, "archive_middle", excluded)
    print("Fresh middle-word extraction", {k: len(v) for k, v in measurements_meta.items()}, flush=True)
    run = "v120-cross-mcs-checks"
    endpoint = read(art(run, "E122", "cross_mcs_checks.json"))
    validation = original_words(art)
    cases = []
    with np.load(art(run, "E122", "cross_mcs_observations.npz"), allow_pickle=False) as data:
        for code in TARGETS:
            edge = layout["marker_systematic_ends"][str(code)]
            sets = ([middle[code]] if code in middle else []) + list(endpoint_words(data, endpoint, code, excluded))
            if code in validation:
                sets.append(validation[code])
            input_families = [(70, i, arr) for i, arr in enumerate(families[70])]
            if code in (70, 88):
                input_families += [(66, i, arr) for i, arr in enumerate(families[66])]
                input_families.append((66, "cancelled_parity_block", cancelled))
            for source, family, original in input_families:
                old_end = SOURCES[source]
                row = translate_rows(original, old_end, edge)
                arm = {"source": source, "family": family, "target": code, "target_systematic_end": edge}
                if row is not None:
                    shifted = translate_rows(shift_systematic(original, old_end), old_end, edge)
                    if shifted is not None:
                        arm.update(support=row[0].tolist(), scores=scores(row, shifted, sets))
                cases.append(arm)
    result = {"schema": "starlink-cross-rate-check-translation/v1", "shared_row_mcs66_family": 4,
              "shared_row_mcs70_family": 0, "translated_mcs66_family": additional[0].tolist(),
              "augmented_rank": rank, "augmented_parity_rank": parity_rank, "cases": cases,
              "archive_middle_words": measurements_meta, "archive_excluded_groups": excluded,
              "archive_split_seed": endpoint["archive_split_seed"], "raw_split_seed": endpoint["raw250_split_seed"],
              "source_archive": {"url": URL, "etag": ETAG, "bytes": SIZE,
                                 "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): s for p, s in inputs.items()}}
    out = experiment_output("E127")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cross_rate_middle_words.npz", **saved)
    np.savez_compressed(out / "mcs66_six_families.npz", row_support=support,
                        check_matrix_packed=np.packbits(matrix, axis=1))
    (out / "cross_rate_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print("Expanded MCS 66 check and parity rank", rank, parity_rank)


if __name__ == "__main__":
    main()
