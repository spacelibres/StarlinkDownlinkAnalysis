"""Exhaustively test folded parity relations when the uncertain prefix is short."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


SEED = 2026112177
REPETITIONS = 199
LIFTS = (128, 256, 512, 1024)
RESIDUES = (1, 2)


def walsh_transform(values):
    values = values.copy()
    half = 1
    while half < len(values):
        pairs = values.reshape(-1, 2, half)
        left, right = pairs[:, 0].copy(), pairs[:, 1].copy()
        pairs[:, 0], pairs[:, 1] = left + right, left - right
        half *= 2
    return values


def fit_dense(keys, output, visible, weights, width):
    output_signs = 1 - 2 * output.astype(float)
    columns = []
    for signs, mask in zip(output_signs.T, visible.T, strict=True):
        if not mask.any():
            columns.append(None)
            continue
        sums = np.bincount(keys[mask], weights=signs[mask] * weights[mask], minlength=1 << width)
        correlation = walsh_transform(sums)
        magnitude = abs(correlation)
        winners = np.flatnonzero(np.isclose(magnitude, magnitude.max(), atol=1e-10, rtol=0))
        columns.append({"coefficients": winners, "positive": correlation[winners] >= 0,
                        "calibration_correlation": float(magnitude.max() / weights[mask].sum())})
    return columns


def fold_word(bits, seen, lift, residue):
    shape = (len(bits), bits.shape[1] // lift, lift // residue, residue)
    folded = np.bitwise_xor.reduce(bits.reshape(shape), axis=2).reshape(len(bits), -1)
    covered = seen.reshape(shape).all(axis=2).reshape(len(bits), -1)
    return folded, covered


def archive_groups(headers, rng):
    parent = list(range((len(headers) + 12) // 13))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    by_header = defaultdict(list)
    for row in headers:
        key = tuple(tuple(row[name]["words_hex"]) for name in ("h1", "h2"))
        by_header[key].append(row["frame"] // 13)
    for family in by_header.values():
        for member in family[1:]:
            parent[root(member)] = root(family[0])
    chunks = np.array([root(i) for i in range(len(parent))])
    groups = np.unique(chunks)
    permuted = rng.permutation(groups)
    fit = sorted(permuted[:len(groups) // 2].tolist())
    return chunks, fit


def group_weights(groups):
    _, inv, count = np.unique(groups, return_inverse=True, return_counts=True)
    return 1 / count[inv]


def frozen_predictions(keys, fits, width):
    sign = np.array([i.bit_count() % 2 for i in range(1 << width)], dtype=bool)
    guesses = np.zeros((len(keys), len(fits)), dtype=bool)
    valid = np.zeros(guesses.shape, dtype=bool)
    for j, chosen in enumerate(fits):
        if chosen is None:
            continue
        candidates = sign[keys[:, None] & chosen["coefficients"][None, :]] ^ ~chosen["positive"]
        guesses[:, j] = candidates[:, 0]
        valid[:, j] = (candidates == candidates[:, :1]).all(axis=1)
    return guesses, valid


def control_origins(groups, rng):
    unique = np.unique(groups)
    indices = [np.flatnonzero(groups == group) for group in unique]
    for _ in range(REPETITIONS):
        order = rng.permutation(len(unique))
        shifted = np.roll(order, int(rng.integers(1, len(unique))))
        positions = np.empty(len(groups), dtype=int)
        for old, new in zip(order, shifted, strict=True):
            positions[indices[old]] = np.resize(indices[new], len(indices[old]))
        yield positions


def score(guess, pred_seen, output, observed, groups):
    masks = observed & pred_seen
    correct = (output == guess) & masks
    support = masks.sum(axis=1)
    successes = correct.sum(axis=1)
    group_scores = []
    for group in np.unique(groups):
        use = groups == group
        amount = int(support[use].sum())
        if amount:
            group_scores.append(float(successes[use].sum() / amount))
    return {"compared": int(support.sum()), "correct": int(successes.sum()), "groups": len(group_scores),
            "group_accuracy": sum(group_scores) / len(group_scores) if group_scores else None}


def audit_model(prefix, prefix_seen, parity, parity_seen, groups, fit_groups, lift, residue, rng):
    in_bits, in_seen = fold_word(prefix, prefix_seen, lift, residue)
    output, observed = fold_word(parity, parity_seen, lift, residue)
    valid = in_seen.all(axis=1)
    training = valid & np.isin(groups, fit_groups)
    held = valid & ~np.isin(groups, fit_groups)
    family = groups[held]
    report = {"lift": lift, "residues": residue, "inputs": in_bits.shape[1], "output_coordinates": output.shape[1],
              "usable_calibration": int(training.sum()), "usable_held": int(held.sum()),
              "calibration_groups": len(np.unique(groups[training])), "held_groups": len(np.unique(family)),
              "models": [], "main": None}
    if min(report["calibration_groups"], report["held_groups"]) < 3:
        return report
    coefficients = in_bits @ (1 << np.arange(in_bits.shape[1]))
    fitted = fit_dense(coefficients[training], output[training], observed[training], group_weights(groups[training]),
                       in_bits.shape[1])
    guess, pred_seen = frozen_predictions(coefficients[held], fitted, in_bits.shape[1])
    report["models"] = [
        None if fit is None else {
            "tied_coefficients": len(fit["coefficients"]),
            "coefficient": int(fit["coefficients"][0]) if len(fit["coefficients"]) == 1 else None,
            "positive": bool(fit["positive"][0]) if len(fit["positive"]) == 1 else None,
            "calibration_correlation": fit["calibration_correlation"],
        } for fit in fitted
    ]
    if not pred_seen.any():
        return report
    output, observed = output[held], observed[held]
    report["main"] = score(guess, pred_seen, output, observed, family)
    report["wrong_group_controls"] = [
        score(guess[origins], pred_seen[origins], output, observed, family)["group_accuracy"]
        for origins in control_origins(family, rng)
    ]
    shifts = rng.integers(1, guess.shape[1], REPETITIONS) if guess.shape[1] > 1 else []
    report["wrong_position_controls"] = [
        score(np.roll(guess, int(shift), axis=1), np.roll(pred_seen, int(shift), axis=1),
              output, observed, family)["group_accuracy"] for shift in shifts
    ]
    return report


def npz_bits(archive, key, length):
    return np.unpackbits(archive[key], axis=-1)[:, :length]


def family_controls(formats, field):
    arms = []
    scores = []
    for label, record in formats.items():
        for key, comparisons in record["arms"].items():
            for arm in comparisons:
                main = arm["main"]
                if main is None or main["group_accuracy"] is None:
                    continue
                v = arm[field]
                if len(v) != REPETITIONS or any(x is None for x in v) or np.std(v) < 1e-10:
                    continue
                observed = main["group_accuracy"]
                scores.append((np.array([observed, *v]) - np.mean(v)) / np.std(v))
                arms.append({"label": int(label), "prefix": key, "lift": arm["lift"],
                             "residues": arm["residues"], "accuracy": observed,
                             "compared": main["compared"]})
    stacked = np.asarray(scores)
    strongest = int(stacked[:, 0].argmax())
    maxima = stacked.max(axis=0)
    return {"scored_arms": len(arms), "leading": arms[strongest], "maxima_standardized": maxima.tolist(),
            "scores_equal_or_better_including_observed": int(np.count_nonzero(maxima >= maxima[0]))}


def pn_fold_detection(inputs):
    run = "v074-repeat-consensus-erasure-final"
    rows = read(checked_artifact(run, "E076", "repeat_consensus.json", inputs))["groups"]
    source = checked_artifact(run, "E076", "repeat_consensus.npz", inputs)
    label_file = checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs)
    model = read(label_file)["frozen_pn_model"]
    pn = equality_predictions(np.arange(16385), pn_bits(), model)
    windows = []
    with np.load(source, allow_pickle=False) as p:
        for group in rows:
            for word in group["triplets"]:
                if "frozen_pn15_control" not in word["leave_one_out"][0]:
                    continue
                key = "_".join(map(str, [*word["frames"], word["resource"]]))
                end = word["region_stop"]
                selected = slice(end - 512, end)
                bits = np.unpackbits(p[f"{key}_bits"])[selected]
                mask = np.unpackbits(p[f"{key}_mask"])[selected].astype(bool)
                bit_errors = bits ^ pn[selected]
                arms = []
                for lift in (16, 32, 64, 128, 256, 512):
                    for residue in RESIDUES:
                        errors, visible = fold_word(bit_errors[None], mask[None], lift, residue)
                        arms.append({"bits_per_block": lift, "residues": residue, "total": visible.size,
                                     "visible": int(visible.sum()),
                                     "observed_parity_errors": int(np.count_nonzero(errors & visible))})
                windows.append({"label": word["label"], "frames": word["frames"], "resource": word["resource"],
                                "observed_bits": int(mask.sum()), "bit_errors": int((bit_errors & mask).sum()),
                                "folds": arms})
    return windows


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--pn-diagnostics"]):
        raise ValueError("Only --pn-diagnostics is supported")
    diagnostics = bool(sys.argv[1:])
    inputs = {}
    seed = np.random.default_rng(SEED)
    headers = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))["frames"]
    chunks, fit_groups = archive_groups(headers, seed)
    run = "v072-padding-parity-final"
    meta = read(checked_artifact(run, "E074", "padding_parity_transfer.json", inputs))["formats"]
    geometry = read(checked_artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json", inputs))
    lengths = defaultdict(set)
    for frame in geometry["word_windows_and_onsets"]:
        for allocation in frame["allocations"]:
            lengths[str(allocation["label"])].add(allocation["width_resources"] * allocation["bits_per_symbol"])
    bits_path = checked_artifact(run, "E074", "padding_prefix_observations.npz", inputs)
    results = {}
    with np.load(bits_path, allow_pickle=False) as archive:
        for label, data in meta.items():
            groups = chunks[[row["frame"] // 13 for row in data["words"]]]
            edge = data["words"][0]["edge"]
            length, = lengths[label]
            if length % 128 not in (0, 1):
                raise ValueError(f"Unexpected label packing for {label}: {length}")
            parity_width = (length // 128) * 128 - edge
            parity = npz_bits(archive, f"{label}_parity", parity_width)
            parity_seen = npz_bits(archive, f"{label}_parity_mask", parity_width).astype(bool)
            results[label] = {"words": data["words"], "group_ids": groups.tolist(), "arms": {},
                              "coded_length_assumed": edge + parity_width,
                              "unused_label_bits": length % 128}
            for key in ("prefix", "conditional_prefix"):
                prefix = npz_bits(archive, f"{label}_{key}", 1024)
                prefix_seen = npz_bits(archive, f"{label}_{key}_mask", 1024).astype(bool)
                arms = []
                for lift in LIFTS:
                    if edge % lift or parity_width % lift:
                        continue
                    for residue in RESIDUES:
                        arms.append(audit_model(prefix, prefix_seen, parity, parity_seen, groups, fit_groups,
                                                lift, residue, np.random.default_rng(SEED)))
                results[label]["arms"][key] = arms
            print("High-order format", label, "words", len(groups), "groups", len(np.unique(groups)), flush=True)
    source5 = "v022-ut-replay"
    code5_path = checked_artifact(source5, "E014", "ut_observations.npz", inputs)
    words = read(checked_artifact(source5, "E014", "ut_transfer.json", inputs))["words"]
    with np.load(code5_path, allow_pickle=False) as archive:
        bits = archive["hard_code5_words"]
    frames = np.array([row["candidate_index"] for row in words])
    training = sorted(seed.permutation(np.unique(frames))[:len(np.unique(frames)) // 2].tolist())
    control = {
        "training_frames": training,
        "arms": [audit_model(bits[:, :1024], np.ones((len(bits), 1024), bool), bits[:, 1024:],
                             np.ones((len(bits), 1024), bool), frames, training, lift, residue,
                             np.random.default_rng(SEED))
                 for lift in LIFTS for residue in RESIDUES],
    }
    result = {
        "schema": "starlink-dense-padding-folds/v1", "seed": SEED, "control_repetitions": REPETITIONS,
        "input_hashes": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "archive_chunk_group_ids": chunks.tolist(), "calibration_groups": fit_groups,
        "code5_positive_control": control, "high_order_formats": results,
    }
    if diagnostics:
        result["family_controls"] = {field: family_controls(results, field)
                                     for field in ("wrong_group_controls", "wrong_position_controls")}
        result["direct_pn_detection"] = pn_fold_detection(inputs)
        result["input_hashes"] = {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()}
    out = experiment_output("E077")
    out.mkdir(parents=True, exist_ok=True)
    (out / "dense_padding_folds.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()
