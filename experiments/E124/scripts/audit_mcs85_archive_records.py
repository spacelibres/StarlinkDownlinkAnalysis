"""Transfer frozen MCS 85 record tests to the full measured Qin archive."""

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations, pairwise
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_gain_transfer import decisions
from audit_mcs85_records import SYSTEMATIC_ENDS, check_frame, detect_records, system_bytes
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


CODE = 85
BITS = SYSTEMATIC_ENDS[CODE]
SERIAL_LANES = 5
NEW_TAG = "03e9"
NEW_DISTANCE = 1512


def link_tags(rows, lengths):
    """Link exact same-tag starts through a continuous verified-boundary path."""
    history, pairs = {}, []
    for i, (first, second) in enumerate(pairwise(rows)):
        label = first["label_hex"]
        if second["byte_offset"] - first["byte_offset"] != lengths.get(label):
            history.clear()
            continue
        if label in history:
            previous = history[label]
            pairs.append((previous, i, i - previous))
        history[label] = i
    return pairs


def counter_pairs(rows, lengths):
    output = []
    for previous, current, ordinal in link_tags(rows, lengths):
        a, b = rows[previous], rows[current]
        left, right = a["candidate_counter_be"], b["candidate_counter_be"]
        output.append({"label": a["label_hex"], "first_offset": a["byte_offset"],
                       "second_offset": b["byte_offset"], "first": left, "second": right,
                       "tag_and_counter_observed": all(r["label_counter_observed_bits"] == 32 for r in (a, b)),
                       "tag_step": (right - left) % 65536, "all_record_step": ordinal,
                       "byte_swapped_step": (b["candidate_counter_le"] - a["candidate_counter_le"]) % 65536,
                       "low_byte_step": (right - left) & 255,
                       "wrap_candidate": (left & 255) == 255})
    return output


def candidate_lattice(rows):
    by_phase = defaultdict(list)
    for row in rows:
        if row["label_hex"] == NEW_TAG and row["label_counter_observed_bits"] == 32:
            by_phase[row["byte_offset"] % NEW_DISTANCE].append(row)
    result = []
    for phase, local in sorted(by_phase.items()):
        mismatches = Counter()
        for left, right in combinations(local, 2):
            step = (right["byte_offset"] - left["byte_offset"]) // NEW_DISTANCE
            first, last = left["candidate_counter_be"], right["candidate_counter_be"]
            for name, offset in (("forward", step), ("reverse", -step), ("double", 2 * step),
                                 ("constant", 0), ("always_one", 1)):
                mismatches[name] += (((first + offset) % 65536) ^ last).bit_count()
        result.append({"phase": phase, "records": len(local), "pairs": len(local) * (len(local) - 1) // 2,
                       "counter_mismatches": mismatches})
    return result


def allocation_records(data, occupancy, pn, wrong_pn, masks, fit):
    observed = system_bytes(data, BITS, pn).ravel()
    wrong = system_bytes(data, BITS, wrong_pn).ravel()
    seen = np.packbits(occupancy[:, :BITS], axis=1, bitorder="little").ravel()
    maximum = fit["detection_max_errors"]
    records = detect_records(observed, *masks, max_distance=maximum)
    for row in records:
        start = row["byte_offset"]
        row["observed_mask_hex"] = seen[start:start + len(masks[0])].tobytes().hex()
        row["label_counter_observed_bits"] = sum(int(x).bit_count() for x in seen[start + 4:start + 8])
        row["label_observed_bits"] = sum(int(x).bit_count() for x in seen[start + 4:start + 6])
    wrong_rows = detect_records(wrong, *masks, max_distance=maximum)
    lengths = fit["fitted_distances_by_exact_label"]
    checked, controls, _ = check_frame(records, len(observed), lengths)
    exploratory, _, _ = check_frame(records, len(observed), {NEW_TAG: NEW_DISTANCE})
    return {"records": records, "wrong_pn_records": wrong_rows, "predictions": checked, "controls": controls,
            "counter_pairs": counter_pairs(records, {**lengths, NEW_TAG: NEW_DISTANCE}),
            "exploratory_predictions": exploratory, "exploratory_tag_lattices": candidate_lattice(records)}


def archive_scan(source, gain, layout, labels, fit, models):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    prefix = np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)]
    signs = 1 - 2 * encode_block(prefix)[:48].astype(float)
    points = np.array([complex(p["i"], p["q"]) for p in labels["cross32_mapping"]])
    points /= max(abs(points))
    mapping = labels_as_bits(np.array([p["label"] for p in labels["cross32_mapping"]]), SERIAL_LANES)
    fit_pn = fit["pn_model"]
    if fit_pn != labels["frozen_pn_model"]:
        raise ValueError("E030 and E052 PN references disagree")
    taps = pn_bits()
    pn = equality_predictions(np.arange(BITS), taps, fit_pn)
    wrong = equality_predictions(np.arange(BITS), taps, {**fit_pn, "phase": fit_pn["phase"] + 1})
    masks = [np.frombuffer(bytes.fromhex(fit[f"{key}_hex"]), np.uint8)
             for key in ("template", "detection_mask", "verification_mask")]

    @lru_cache(maxsize=1)
    def block(first):
        stop = min(first + source.chunks[0], len(source))
        return as_complex(source[first:stop, 1:, :])[..., NONPILOT % 1024]

    for row, shape in zip(gain["frames"], layout["word_windows_and_onsets"], strict=True):
        frame = row["frame"]
        if frame != shape["frame"]:
            raise ValueError("Archive frame order changed")
        targets = []
        for obs, alloc in zip(row["allocations"], shape["allocations"], strict=True):
            if (obs["label"], obs["start"]) != (alloc["label"], alloc["start"]):
                raise ValueError("Archive allocation disagrees with gain receipt")
            if alloc["label"] == CODE:
                targets.append(alloc)
        if not targets:
            continue
        first = frame // source.chunks[0] * source.chunks[0]
        flat = remove_header_gain(block(first)[frame - first] * rotation, signs)
        axis = complex(*row["axis"])
        for alloc in targets:
            low, count, width = (alloc[k] for k in ("start", "count", "width_resources"))
            samples = (flat[low:low + count * width] / axis).reshape(count, width)
            hard = mapping[nearest_symbols(samples, points)].reshape(count, -1)
            exact, seen, _, _ = decisions(flat, axis, alloc, models, BITS)
            if np.any(hard[seen] != exact[seen]):
                raise ValueError("Nearest and exact-lattice lanes disagree at a visible point")
            yield {"frame": frame, "group": row["group"], "calibration": row["calibration"],
                   "resource_start": low, "words": count, "resources_per_word": width,
                   **allocation_records(hard, seen, pn, wrong, masks, fit)}


def tally(rows):
    labels = defaultdict(Counter)
    control = defaultdict(Counter)
    for row in rows:
        for claim in row["predictions"] + row["exploratory_predictions"]:
            labels[claim["label"]][claim["status"]] += 1
        for other in row["controls"]:
            key = other["type"]
            if key == "permutation":
                key = str(other["distances"])
            if key == "constant":
                key = f"constant_{next(iter(other['distances'].values()))}"
            control[key]["matched"] += other["matches"]
            control[key]["compared"] += other["compared"]
    return {"allocations": len(rows), "frames": len({r["frame"] for r in rows}),
            "words": sum(r["words"] for r in rows), "groups": sorted({r["group"] for r in rows}),
            "records": sum(len(r["records"]) for r in rows),
            "wrong_pn_records": sum(len(r["wrong_pn_records"]) for r in rows),
            "predictions": labels, "controls": control}


def narrow_projection(rows, windows):
    targets = []
    for allocation in rows:
        for row in allocation["records"]:
            # Tag and counter at offsets 4–7, including a possible codeword crossing.
            first = (row["byte_offset"] + 4) * 8
            words, bits = np.divmod(np.arange(first, first + 32), BITS)
            resources = allocation["resource_start"] + words * allocation["resources_per_word"] + bits // SERIAL_LANES
            targets.append(NONPILOT[resources % len(NONPILOT)])
    result = []
    for window in windows:
        seen = np.isin(np.asarray(targets).reshape(-1, 32), window["carriers"]).sum(axis=1)
        result.append({"kind": window["kind"], "labels": window["labels"],
                       "bits_per_tag_and_counter": np.bincount(seen, minlength=33).tolist()})
    return result


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    fit = read(artifact("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    gain = read(artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    layout = read(artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json"))
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    models = constellation_models(labels, inputs)
    binding = read(artifact("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json"))["overlap"]["pairs"]
    excluded = {gain["frames"][pair["qin_frame"]]["group"] for pair in binding}
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        rows = list(archive_scan(archive["yDataDec"], gain, layout, labels, fit, models))
    held = [row for row in rows if not row["calibration"] and row["group"] not in excluded]
    coverage = read(artifact("v120-cross-mcs-checks", "E122", "cross_mcs_checks.json"))["leo"]
    result = {"schema": "starlink-mcs85-full-record-scan/v1", "archive_group_seed": 2026112177,
              "archive_overlapping_groups_excluded": sorted(excluded), "frozen_mapping": fit["fitted_distances_by_exact_label"],
              "exploratory_candidate": {NEW_TAG: NEW_DISTANCE}, "rows": rows,
              "held_without_overlap": tally(held),
              "held_by_group": {group: tally([r for r in held if r["group"] == group]) for group in sorted({r["group"] for r in held})},
              "all": tally(rows), "leo_ideal_projection": narrow_projection(held, coverage),
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
              "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                                 "pages": [reader.receipts[k] for k in sorted(reader.receipts)]}}
    output = experiment_output("E124")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mcs85_archive_records.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(len(rows), "allocations,", result["all"]["records"], "records;", len(held), "held allocations", flush=True)
    print("Held mapping:", result["held_without_overlap"]["predictions"], flush=True)


if __name__ == "__main__":
    main()
