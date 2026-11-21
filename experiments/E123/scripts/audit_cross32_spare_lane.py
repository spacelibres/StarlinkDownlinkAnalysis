"""Measure the extra cross-32QAM lane and test independent word rounding."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH, CacheOnly, constellation_models
from audit_code5_gain_transfer import decisions, four_corner_slots
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


WIDTH = 5
RESOURCES = (BIT_LENGTH + WIDTH - 1) // WIDTH
LANE = WIDTH * RESOURCES - 1
RAW_CODES = (71, 78, 85, 88, 92)
STRONG_MARGIN = 0.08
PN_GUARD = 128


def joint_lane(word):
    last_resource = ((word + 1) * BIT_LENGTH + WIDTH - 1) // WIDTH - 1
    return WIDTH * (last_resource - word * RESOURCES + 1) - 1


def paired_zero_counts(target, comparison, observed):
    return np.bincount(2 * target[observed] + comparison[observed], minlength=4).reshape(2, 2).tolist()


def observe(source, meta, origin, overlapping_groups):
    result = []
    for key, cohort in meta[origin].items():
        label, role = key.split("_")
        if int(label) in (66, 70):
            continue
        bits, seen = (np.unpackbits(source[f"{origin}_{key}_{k}"], axis=1) for k in ("bits", "seen"))
        for row, values, mask in zip(cohort["rows"], bits, seen, strict=True):
            joint = joint_lane(row["word"])
            if not 0 <= joint <= LANE:
                raise ValueError("Continuous-model position is outside the saved word")
            positions = {"final": LANE, "previous": LANE - WIDTH,
                         "same_subcarrier_previous_slot": LANE - WIDTH * len(NONPILOT),
                         "joint": joint}
            record = {k: int(values[p]) if mask[p] else None for k, p in positions.items()}
            result.append({**row, "role": role,
                           "overlap_group": row["group"] in overlapping_groups,
                           "joint_position_is_distinct": joint != LANE, **record})
    return result


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    by_group = {}
    for group, group_rows in sorted(groups.items()):
        result = {"final_seen": sum(row["final"] is not None for row in group_rows),
                  "final_ones": sum(row["final"] == 1 for row in group_rows)}
        for control in ("previous", "same_subcarrier_previous_slot", "joint"):
            usable = [r for r in group_rows if r["final"] is not None and r[control] is not None
                      and (control != "joint" or r["joint_position_is_distinct"])]
            target = np.array([r["final"] for r in usable], dtype=np.uint8)
            reference = np.array([r[control] for r in usable], dtype=np.uint8)
            result[control] = paired_zero_counts(target, reference, np.ones(len(usable), dtype=bool))
        by_group[group] = result
    return {"words": len(rows), "groups": by_group,
            **{k: sum(g[k] for g in by_group.values()) for k in ("final_seen", "final_ones")},
            "comparisons": {
                k: np.sum([g[k] for g in by_group.values()], axis=0, dtype=int).tolist() if groups else [[0, 0], [0, 0]]
                for k in ("previous", "same_subcarrier_previous_slot", "joint")
            }}


def bit_margin(samples, points, is_one):
    distances = abs(samples[..., None] - points) ** 2
    return distances[..., is_one].min(axis=-1) - distances[..., ~is_one].min(axis=-1)


def characterize(values):
    return {"seen": len(values), "ones": int(np.count_nonzero(values < 0)),
            "strong_ones": int(np.count_nonzero(values < -STRONG_MARGIN))}


def observations_from_iq(data, meta, labels):
    point_labels = labels["cross32_mapping"]
    points = np.array([complex(p["i"], p["q"]) for p in point_labels]) / np.sqrt(20)
    is_one = np.array([bool(p["label"] & 1) for p in point_labels])
    if not np.allclose(points, data["cross32_points"]):
        raise ValueError("Frozen constellation order disagrees with E030")
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"])
    margins = {code: bit_margin(data[f"mcs{code}_symbols"][:, -1], points, is_one)
               for code in RAW_CODES}
    intervals = defaultdict(list)
    for row in labels["word_results"]:
        code = row["format"]
        if code not in RAW_CODES or not row["accepted"]:
            continue
        samples = data[f"mcs{code}_symbols"][row["candidate_index"]]
        first = (row["start"] + PN_GUARD + WIDTH - 1) // WIDTH
        indices = np.arange(first, (row["stop"] - PN_GUARD) // WIDTH)
        # Use only symbols fully inside the guarded known-PN span.
        indices = indices[pn[WIDTH * indices + WIDTH - 1] == 0]
        intervals[row["frame"]].extend(bit_margin(samples[indices], points, is_one).tolist())
    offset = defaultdict(int)
    records = []
    for region in meta["regions"]:
        code, frame = region["format"], region["frame"]
        if code not in RAW_CODES:
            continue
        index, count = offset[code], region["count"]
        if not np.all(data[f"mcs{code}_frames"][index:index + count] == frame):
            raise ValueError("Raw region order disagrees with E030")
        for i in range(count):
            value = margins[code][index + i]
            records.append({"label": code, "frame": frame,
                            "calibration": frame in meta["calibration_frames"],
                            "word": i, "region_count": count, "start": region["start"],
                            "resource": region["start"] + i * RESOURCES + RESOURCES - 1,
                            "margin": float(value), "bit": int(value < 0)})
        offset[code] += count
    if any(offset[code] != len(margins[code]) for code in RAW_CODES):
        raise ValueError("Unaccounted raw word")
    by_frame = {}
    for frame in sorted({r["frame"] for r in records}):
        group = np.array([r["margin"] for r in records if r["frame"] == frame])
        by_frame[frame] = {"calibration": frame in meta["calibration_frames"],
                           "extra": characterize(group), "pn_known_zero": characterize(np.array(intervals[frame]))}
    return records, by_frame


def recheck_overlap(data, raw, gain, layout, binding, models):
    qin_by_raw = {r["ut_candidate"]: r["qin_frame"] for r in binding}
    shifts = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    prefix = np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)]
    signs = 1 - 2 * encode_block(prefix)[:48].astype(float)
    by_frame = defaultdict(list)
    for r in raw:
        if r["frame"] in qin_by_raw:
            by_frame[r["frame"]].append(r)
    result = []
    for frame, rows in sorted(by_frame.items()):
        qin = qin_by_raw[frame]
        geometry = layout["word_windows_and_onsets"][qin]
        ref = gain["frames"][qin]
        if geometry["frame"] != qin or ref["frame"] != qin:
            raise ValueError("Qin frame numbering changed")
        allocs = {(a["label"], a["start"], a["count"]): a for a in geometry["allocations"]}
        block = as_complex(data[qin:qin + 1, 1:, :])[0][..., NONPILOT % 1024]
        flat = remove_header_gain(block * shifts, signs)
        axis = complex(*ref["axis"])
        corners = four_corner_slots(flat, axis)
        pairs = {}
        for r in rows:
            key = (r["label"], r["start"], r["region_count"])
            if key not in pairs:
                alloc = allocs[key]
                if alloc["width_resources"] != RESOURCES:
                    raise ValueError("Qin and raw resource geometry differ")
                bits, mask, _, _ = decisions(flat, axis, alloc, models, 0, corners)
                pairs[key] = bits[:, LANE], mask[:, LANE]
            bits, mask = pairs[key]
            word = r["word"]
            result.append({**r, "qin_frame": qin, "qin_group": ref["group"],
                           "archive_bit": int(bits[word]) if mask[word] else None})
    return result


def coverage(rows, windows):
    carriers = NONPILOT[(np.array([r["resource"] for r in rows]) + RESOURCES - 1) % len(NONPILOT)]
    report = []
    for window in windows:
        hits = np.isin(carriers, window["carriers"])
        report.append({"kind": window["kind"], "labels": window["labels"], "visible": int(hits.sum()),
                       "possible": len(rows), "words": [
                           [row["frame"], row["label"], row["resource"] + RESOURCES - 1]
                           for row, hit in zip(rows, hits, strict=True) if hit]})
    return report


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    label_run = "v028-ut-cross32-labels-final"
    labels = read(art(label_run, "E030", "ut_cross32_labels.json"))
    raw_run = "v028-ut-cross32-extract-02"
    raw_meta = read(art(raw_run, "E030", "ut_cross32_extraction.json"))
    with np.load(art(raw_run, "E030", "ut_cross32_symbols.npz"), allow_pickle=False) as data:
        raw, by_frame = observations_from_iq(data, raw_meta, labels)
    binding = read(art("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json"))["overlap"]["pairs"]
    gain = read(art("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    layout = read(art("v068-pn-padding-starts", "E070", "pn_padding_starts.json"))
    overlapping_groups = {gain["frames"][r["qin_frame"]]["group"] for r in binding}
    model = constellation_models(labels, inputs)
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        pairs = recheck_overlap(archive["yDataDec"], raw, gain, layout, binding, model)
    run = "v120-cross-mcs-checks"
    meta = read(art(run, "E122", "cross_mcs_checks.json"))
    with np.load(art(run, "E122", "cross_mcs_observations.npz"), allow_pickle=False) as source:
        archive_rows = observe(source, meta, "archive", overlapping_groups)
        raw_rows = observe(source, meta, "raw250", set())
    # Enforce that both IQ-derived receipts refer to the same serialized decisions.
    by_word = {(r["label"], r["frame"], r["resource"]): r["bit"] for r in raw}
    for row in raw_rows:
        key = (row["label"], row["frame"], row["resource"] + RESOURCES - 1)
        if row["final"] != by_word.pop(key):
            raise ValueError("E122 packed bits differ from the soft-distance sign")
    if by_word:
        raise ValueError("Raw packed rows missing")
    held_archive = [r for r in archive_rows if not r["calibration"] and not r["overlap_group"]]
    held_raw = [r for r in raw_rows if not r["calibration"]]
    results = {
        "schema": "starlink-cross32-spare-lane/v1", "serial_index": LANE,
        "archive_overlapping_groups_excluded": sorted(overlapping_groups),
        "archive_split_seed": meta["archive_split_seed"], "raw250_split_seed": raw_meta["split_seed"],
        "raw_strong_margin": STRONG_MARGIN, "pn_guard_bits": PN_GUARD,
        "archive": {"rows": archive_rows, "held_no_overlap": summarize(held_archive),
                    "by_label_role": {f"{c}_{role}": summarize([r for r in held_archive if r["label"] == c and r["role"] == role])
                                      for c in sorted({r["label"] for r in archive_rows}) for role in ("nonterminal", "terminal")}},
        "raw250": {"rows": raw_rows, "reserved": summarize(held_raw), "margins": raw, "by_frame": by_frame,
                   "same_waveform_overlap": pairs},
        "leo_projection_for_held_qin_origins": coverage(held_archive, meta["leo"]),
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                           "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
    }
    output = experiment_output("E123")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cross32_spare_lane.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, summary in (("Qin held without overlapping groups", results["archive"]["held_no_overlap"]),
                          ("Raw held", results["raw250"]["reserved"])):
        print(name, summary["final_ones"], "/", summary["final_seen"], "ones", flush=True)
        print("Matched comparisons", summary["comparisons"], flush=True)


if __name__ == "__main__":
    main()
