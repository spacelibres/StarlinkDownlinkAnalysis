"""Use a frozen PN15 sequence to learn and transfer a cross-32QAM labeling."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import DATA as E028, collect, equality_predictions, matching_run
from audit_ut_binary_tail import sha256
from extract_ut_cross32 import TARGETS
from receiver_models import pn_bits
from validate_qc_figure import constellation

OBSERVATION_RUN = ROOT / "runs/v028-ut-cross32-extract-02"
OBSERVATION = OBSERVATION_RUN / "artifacts/experiments/E030/results"
PN_RECORD = ROOT / "runs/v027-qam-pn15-final/artifacts/experiments/E029/results/qam_pn15_audit.json"
WINDOW = 512
HOP = 256
FIRST_PREFIX_BYTES = 128


def label_assignment(counts):
    row, col = linear_sum_assignment(counts, maximize=True)
    return col, int(counts[row, col].sum())


def table_for(indices, labels, points):
    return np.bincount(indices * points + labels, minlength=points * points).reshape(points, points)


def labels_as_bits(labels, width):
    return ((labels[..., None] >> np.arange(width - 1, -1, -1)) & 1).astype(np.uint8)


def nearest_symbols(values, points):
    rows = []
    for chunk in np.array_split(values, max(1, (len(values) + 15) // 16)):
        rows.append(np.argmin(abs(chunk[..., None] - points) ** 2, axis=-1))
    return np.concatenate(rows)


def final_word_mask(frames):
    return np.r_[frames[1:] != frames[:-1], True]


def study_mapping(symbols, expected, frames, training):
    """Use at most one strong fitting window from each MCS 85 calibration frame."""
    selected = []
    accumulated = np.zeros((32, 32), dtype=np.int64)
    for i in np.flatnonzero(final_word_mask(frames) & np.isin(frames, training)):
        candidates = []
        for lo in range(0, len(expected) - WINDOW + 1, HOP):
            window = slice(lo, lo + WINDOW)
            counts = table_for(symbols[i, window], expected[window], 32)
            label, score = label_assignment(counts)
            candidates.append((score, lo, counts, label))
        score, lo, count, label = max(candidates, key=lambda x: x[0])
        if score < .7 * WINDOW:
            continue
        accumulated += count
        selected.append({"candidate_index": int(i), "frame": int(frames[i]), "first_symbol": lo,
                         "symbols_compared": WINDOW, "individually_fitted_exact_matches": score,
                         "individual_mapping": label.tolist()})
    if not selected:
        raise ValueError("No training window contains a strong PN15 match")
    mapping, score = label_assignment(accumulated)
    alternatives = []
    for point, label in enumerate(mapping):
        blocked = accumulated.copy()
        blocked[point, label] = -1 - accumulated.sum()
        alternatives.append(label_assignment(blocked)[1])
    return mapping, {
        "selected_mcs85_calibration_windows": selected,
        "summed_confusion": accumulated.tolist(),
        "correct_points_in_calibration": score,
        "calibration_symbols": int(accumulated.sum()),
        "calibration_gap_to_second_bijection": score - max(alternatives),
        "all_targets_have_observations": bool(np.all(accumulated.sum(axis=0) > 0)),
        "all_points_have_observations": bool(np.all(accumulated.sum(axis=1) > 0)),
    }


def predict(pn, fit, n):
    return equality_predictions(np.arange(n), pn, fit)


def cases_for(code, values, frames, points, mapping, pn, fit, training):
    width = 4 if code == 57 else 5
    n = 8192 if code == 57 else 16384
    if values.shape[1] * width < n or values.shape[1] * width - n > 1:
        raise ValueError("Candidate symbol width inconsistent")
    decided = nearest_symbols(values, points)
    all_labels = labels_as_bits(mapping[decided], width).reshape(len(values), -1)
    observation = all_labels[:, :n]
    pn_prefix = predict(pn, fit, n)
    controls = [predict(pn, {**fit, "phase": fit["phase"] + offset}, n) for offset in (-1, 1)]
    word_ids = final_word_mask(frames)
    expected_labels = n // width * width
    v = pn_prefix[:expected_labels].reshape(-1, width) @ (1 << np.arange(width - 1, -1, -1))
    waveform = points[np.argsort(mapping)[v]]
    rows = []
    for i, row in enumerate(observation):
        interval = matching_run(row != pn_prefix)
        wrong = [matching_run(row != control) for control in controls]
        result = {**interval, "format": code, "frame": int(frames[i]), "candidate_index": i,
                  "calibration_frame": bool(frames[i] in training), "allocation_final_word": bool(word_ids[i]),
                  "max_wrong_phase_run_bits": max(x["length"] for x in wrong),
                  "accepted_wrong_phase": any(x["accepted"] for x in wrong)}
        if interval["accepted"]:
            lo, hi = (interval["start"] + width - 1) // width, interval["stop"] // width
            result["match_waveform_mse"] = float(np.mean(abs(values[i, lo:hi] - waveform[lo:hi]) ** 2))
            result["matching_hard_symbols"] = int(np.count_nonzero(decided[i, lo:hi] == np.argsort(mapping)[v[lo:hi]]))
            result["matching_region_complete_symbols"] = hi - lo
        rows.append(result)
    spare = all_labels[:, n:]
    extra = {"observed_bits": spare.size, "ones": int(spare.sum())}
    if spare.size:
        last_two = labels_as_bits(mapping[decided[:, -2:]], width)
        costs = abs(values[:, -1, None] - points[None, :]) ** 2
        margin = costs[:, (mapping & 1) == 1].min(axis=1) - costs[:, (mapping & 1) == 0].min(axis=1)
        extra.update({
            "final_symbol_ones_by_lane": last_two[:, 1].sum(axis=0).tolist(),
            "penultimate_symbol_ones_by_lane": last_two[:, 0].sum(axis=0).tolist(),
            "signed_zero_margin_quantiles": np.quantile(margin, [0., .01, .05, .5, 1.]).tolist(),
            "strong_ones_margin_gt_point08": int(np.count_nonzero(margin < -.08)),
        })
    return rows, extra


def prefix_diagnostics(e028_source, pn, fit):
    reference = json.loads((E028 / "ut_qam_extraction.json").read_text(encoding="utf-8"))
    earlier, _, _ = collect(e028_source, reference)
    mask = predict(pn, fit, FIRST_PREFIX_BYTES * 8)
    results = []
    for code, (bits, _, _, records) in earlier.items():
        train = np.array([r["calibration"] and not r["last_word_of_allocation"] for r in records])
        held = np.array([not r["calibration"] and not r["last_word_of_allocation"] for r in records])
        data = np.packbits(bits[:, :len(mask)] ^ mask, axis=1, bitorder="little")
        counts = np.stack([np.bincount(x, minlength=256) for x in data[train].T])
        choices = counts.argmax(axis=1)
        held_matches = np.count_nonzero(data[held] == choices, axis=0)
        results.append({"format": code, "prefix_bytes_tested": FIRST_PREFIX_BYTES,
                        "calibration_words": int(train.sum()), "reserved_words": int(held.sum()),
                        "largest_calibration_modal_hits": int(counts.max()),
                        "largest_reserved_hits_of_calibration_modes": int(held_matches.max()),
                        "reserved_hits_of_calibration_modes_by_coordinate": held_matches.tolist()})
    return results


def main():
    require_run_context()
    observations = OBSERVATION / "ut_cross32_symbols.npz"
    metadata = OBSERVATION / "ut_cross32_extraction.json"
    e028_source = E028 / "ut_qam_symbols.npz"
    files = (observations, metadata, OBSERVATION_RUN / "manifest.json", PN_RECORD, e028_source,
             E028 / "ut_qam_extraction.json")
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    receipt = json.loads(files[2].read_text(encoding="utf-8"))["changed_artifacts"]
    for path in (observations, metadata):
        if hashes[path.relative_to(ROOT).as_posix()] != receipt[f"experiments/E030/results/{path.name}"]["sha256"]:
            raise ValueError("Input symbol archive changed")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    calibrate = meta["calibration_frames"]
    fit = json.loads(PN_RECORD.read_text(encoding="utf-8"))["fit"]["winner"]
    pn = pn_bits()
    expected = predict(pn, fit, 3276 * 5).reshape(-1, 5) @ (1 << np.arange(4, -1, -1))
    rows, extra = [], {}
    points16, gray = constellation(39)
    map16 = gray @ (1 << np.arange(3, -1, -1))
    with np.load(observations, allow_pickle=False) as z:
        points = z["cross32_points"]
        indices = nearest_symbols(z["mcs85_symbols"], points)
        mapping, info = study_mapping(indices, expected, z["mcs85_frames"], calibrate)
        for code in TARGETS:
            alphabet, label = (points16, map16) if code == 57 else (points, mapping)
            results, lane_counts = cases_for(code, z[f"mcs{code}_symbols"], z[f"mcs{code}_frames"],
                                              alphabet, label, pn, fit, calibrate)
            rows.extend(results)
            extra[str(code)] = lane_counts
            print(code, "words", len(results), "PN matches", sum(r["accepted"] for r in results), flush=True)
    report = {
        "schema": "starlink-ut-cross32-labels/v1", "inputs_sha256": hashes, "frozen_pn_model": fit,
        "frame_split": {g: meta[g + "_frames"] for g in ("calibration", "reserved")},
        "training_code": 85,
        "mapping_calibration": info,
        "cross32_mapping": [{"i": int(round(x.real * np.sqrt(20))), "q": int(round(x.imag * np.sqrt(20))), "label": int(m)}
                            for x, m in zip(points, mapping, strict=True)],
        "candidate_unused_lane_counts_outside_16384": extra,
        "word_results": rows,
        "preceding_66_70_fixed_prefix_diagnostic": prefix_diagnostics(e028_source, pn, fit),
        "scope": "Receiver-relative hard bit labeling from the repeated PN15 segment. No FEC, BER against independent bit truth, or application fields established.",
    }
    out = experiment_output("E030")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ut_cross32_labels.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
