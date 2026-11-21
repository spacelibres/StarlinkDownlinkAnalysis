"""Test a recurring nonzero pattern under the frozen UT PN15 model."""

import json
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import collect, equality_predictions, matching_run
from audit_ut_binary_tail import sha256
from audit_ut_cross32_labels import final_word_mask, labels_as_bits, nearest_symbols
from receiver_models import pn_bits
from validate_qc_figure import constellation

QAM16_RUN = "v026-ut-qam-extract-repeat"
CROSS32_RUN = "v028-ut-cross32-extract-02"
LABEL_RUN = "v028-ut-cross32-labels-final"
PN_RUN = "v027-qam-pn15-final"
CANDIDATE_STOPS = {57: 5376, 66: 12288, 70: 12800, 71: 10752,
                   78: 11520, 85: 12288, 88: 12800, 92: 13056}
FIT_WORD = 66
FIT_SPAN = 512
FIT_WIDTH = 192
MIN_RUN = 144
MODELS = ("frozen_pn15", "raw", "pn_phase_minus1", "pn_phase_plus1")


class WordObservations(NamedTuple):
    bits: np.ndarray
    frames: np.ndarray
    symbols: np.ndarray
    points_by_label: np.ndarray


def bit_weights(width):
    return 1 << np.arange(width - 1, -1, -1)


def sort_points(points, labels):
    return points[np.argsort(labels @ bit_weights(labels.shape[1]))]


def fundamental_period(motif):
    indices = np.arange(len(motif))
    return next(period for period in range(1, len(motif) + 1)
                if len(motif) % period == 0 and np.array_equal(motif, motif[indices % period]))


def learn_pattern(words, frames):
    candidates = []
    for bits, frame in zip(words, frames, strict=True):
        for width in range(8, 65, 8):
            for start in range(0, FIT_SPAN - FIT_WIDTH + 1, 8):
                indices = np.arange(start, start + FIT_WIDTH) % width
                ones = np.bincount(indices, weights=bits[start:start + FIT_WIDTH], minlength=width)
                counts = np.bincount(indices, minlength=width)
                motif = (2 * ones > counts).astype(np.uint8)
                differences = int(np.count_nonzero(bits[start:start + FIT_WIDTH] != motif[indices]))
                score = ((FIT_WIDTH - differences) * np.log(.97 / .5)
                         + differences * np.log(.03 / .5) - width * np.log(2))
                candidates.append((float(score), differences, int(frame), start, width, motif))
    candidates.sort(key=lambda row: -row[0])
    best_by_width = [next(row for row in candidates if row[4] == width) for width in range(8, 65, 8)]
    return candidates[0][-1], [
        {"penalized_score": score, "mismatches": differences, "frame": frame, "window_start": start,
         "window_length": FIT_WIDTH, "template_bits": width}
        for score, differences, frame, start, width, _ in best_by_width
    ]


def scan_pattern(bits, motif):
    positions = np.arange(len(bits))
    rows = [matching_run(bits != motif[(positions + shift) % len(motif)])
            for shift in range(0, len(motif), 8)]
    byte_phase, peak = max(enumerate(rows), key=lambda pair: pair[1]["log_score_gain"])
    start, stop, length = (peak[k] for k in ("start", "stop", "length"))
    errors = peak["mismatches"]
    return {"start": start, "stop": stop, "length": length, "mismatches": errors,
            "score": peak["log_score_gain"], "byte_phase": byte_phase,
            "strict": length >= MIN_RUN and 10 * errors <= length,
            "exploratory": length >= MIN_RUN and 5 * errors <= length}


def score_waveform(values, points_by_label, width, region, motif, masks):
    lower = (region["start"] + width - 1) // width
    upper = region["stop"] // width
    positions = np.arange(lower * width, upper * width)
    source = motif[(positions + 8 * region["byte_phase"]) % len(motif)]
    selected = values[lower:upper]
    scores = {}
    scrambled = {name: source ^ mask[positions] for name, mask in masks.items()}
    scrambled["pn_only"] = masks["frozen_pn15"][positions]
    for name, prediction in scrambled.items():
        labels = prediction.reshape(-1, width) @ bit_weights(width)
        predicted = points_by_label[labels]
        scores[name] = float(np.mean(abs(selected - predicted) ** 2))
    nearest_mse = float(np.mean(np.min(abs(selected[:, None] - points_by_label[None, :]) ** 2, axis=1)))
    return {"fully_contained_symbols": len(selected), "mean_squared_distance": scores,
            "nearest_point_mean_squared_distance": nearest_mse}


def read(path):
    return json.loads(path.read_bytes())


def checked_artifact(run, experiment, name, inputs):
    root = ROOT / "runs" / run
    manifest_path = root / "manifest.json"
    if manifest_path not in inputs:
        inputs[manifest_path] = sha256(manifest_path)
    manifest = read(manifest_path)
    if manifest["returncode"] or manifest["canonical_evidence_mutations"]:
        raise ValueError(f"Unsuccessful or mutated input run {run}")
    key = f"experiments/{experiment}/results/{name}"
    path = root / "artifacts" / key
    actual = sha256(path)
    if actual != manifest["changed_artifacts"][key]["sha256"]:
        raise ValueError(f"Input differs from manifest: {path}")
    inputs[path] = actual
    return path


def observations(inputs):
    meta16 = checked_artifact(QAM16_RUN, "E028", "ut_qam_extraction.json", inputs)
    qam16 = checked_artifact(QAM16_RUN, "E028", "ut_qam_symbols.npz", inputs)
    cross32 = checked_artifact(CROSS32_RUN, "E030", "ut_cross32_symbols.npz", inputs)
    label_source = checked_artifact(LABEL_RUN, "E030", "ut_cross32_labels.json", inputs)
    pn_source = checked_artifact(PN_RUN, "E029", "qam_pn15_audit.json", inputs)
    split16 = read(meta16)
    pn_record = read(pn_source)
    label_record = read(label_source)
    fit = pn_record["fit"]["winner"]
    if fit != label_record["frozen_pn_model"]:
        raise ValueError("Archived PN models disagree")
    frame_split = label_record["frame_split"]
    if any(frame_split[name] != split16[f"{name}_frames"] for name in ("calibration", "reserved")):
        raise ValueError("Archived frame partitions disagree")
    result = {}
    archived16, pts16, gray16 = collect(qam16, split16)
    lut16 = sort_points(pts16, gray16)
    for code, (bits, _, symbols, meta) in archived16.items():
        result[code] = WordObservations(bits, np.array([row["frame"] for row in meta]), symbols, lut16)
    with np.load(cross32, allow_pickle=False) as archive:
        map32 = labels_as_bits(np.array([row["label"] for row in label_record["cross32_mapping"]]), 5)
        pts16, map16 = constellation(39)
        pts32 = archive["cross32_points"]
        for code in (57, 71, 78, 85, 88, 92):
            points, labels = (pts16, map16) if code == 57 else (pts32, map32)
            frames = archive[f"mcs{code}_frames"]
            values = archive[f"mcs{code}_symbols"]
            decided = nearest_symbols(values, points)
            bits = labels[decided].reshape(len(frames), -1)
            result[code] = WordObservations(bits[:, :8192 if code == 57 else 16384], frames,
                                            values, sort_points(points, labels))
    return result, frame_split, fit


def main():
    require_run_context()
    inputs = {}
    datasets, split, pn_fit = observations(inputs)
    phase_codes = (pn_fit, {**pn_fit, "phase": pn_fit["phase"] - 1},
                   {**pn_fit, "phase": pn_fit["phase"] + 1})
    indices = np.arange(16384)
    pn, minus, plus = (equality_predictions(indices, pn_bits(), model) for model in phase_codes)
    masks = dict(zip(MODELS, (pn, np.zeros_like(pn), minus, plus), strict=True))
    train, frames, _, _ = datasets[FIT_WORD]
    first = np.r_[True, frames[1:] != frames[:-1]]
    fit_rows = first & np.isin(frames, split["calibration"])
    motif, candidates = learn_pattern(train[fit_rows, :FIT_SPAN] ^ pn[:FIT_SPAN], frames[fit_rows])
    rows = []
    for code, (bits, group, values, points_by_label) in sorted(datasets.items()):
        beginning = np.r_[True, group[1:] != group[:-1]]
        end = final_word_mask(group)
        for index, (word, frame, first_word, final_word) in enumerate(zip(bits, group, beginning, end, strict=True)):
            scores = {key: scan_pattern(word ^ mask[:len(word)], motif) for key, mask in masks.items()}
            row = {"format": code, "frame": int(frame), "candidate": index,
                   "frame_partition": "calibration" if frame in split["calibration"] else "reserved",
                   "first": bool(first_word), "final": bool(final_word),
                   "candidate_systematic_stop": CANDIDATE_STOPS[code], "models": scores}
            if scores["frozen_pn15"]["exploratory"]:
                width = len(points_by_label).bit_length() - 1
                row["waveform"] = score_waveform(values[index], points_by_label, width,
                                                  scores["frozen_pn15"], motif, masks)
            rows.append(row)
    out = experiment_output("E031")
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "starlink-ut-repeated-data/v1",
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "frame_split": split, "inherited_split_seed": 2026112128, "pn_model": pn_fit,
        "fit_format": FIT_WORD, "fit_first_words_only": True, "fit_candidates": candidates,
        "template_absolute_phase_lsb_first_hex": np.packbits(motif, bitorder="little").tobytes().hex(),
        "fundamental_period_bits": fundamental_period(motif),
        "minimum_run_bits": MIN_RUN, "strict_max_error": .1, "exploratory_max_error": .2,
        "rows": rows,
        "scope": "An exploratory same-capture structural test across whole-frame groups, with word-relative "
                 "coordinates and a fixed template. The symbol decisions do not establish payload semantics.",
    }
    (out / "ut_repeated_data.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for model in MODELS:
        counts = [sum(row["models"][model][key] for row in rows) for key in ("strict", "exploratory")]
        print(model, len(rows), "words;", *counts, "strict and exploratory")


if __name__ == "__main__":
    main()
