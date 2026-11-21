"""Search low-weight affine relations on the high-order candidate parity suffix."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_dense_padding_folds import REPETITIONS, control_origins
from audit_gain_calibrated_folds import EXTRACTION
from audit_ut_repeated_data import checked_artifact, read


SEED = 2026112182
TRIALS = 4
MIN_GROUPS = 8
FIT_MINIMUM = .94
FAMILY_LIMIT_PER_WEIGHT = 512


def exact_candidates(bits, seen):
    """Find weight-two or weight-three affine equalities on complete, varying columns."""
    if not 2 <= len(bits) <= 63:
        raise ValueError("Expected 2 to 63 registration words")
    weight = bits.sum(axis=0)
    positions = np.flatnonzero(seen.all(axis=0) & (weight > 0) & (weight < len(bits)))
    signatures = bits[:, positions].astype(np.uint64).T @ (np.uint64(1) << np.arange(len(bits), dtype=np.uint64))
    complement = np.uint64((1 << len(bits)) - 1)
    canonical = np.minimum(signatures, signatures ^ complement)
    buckets = defaultdict(list)
    for index, value in enumerate(canonical):
        buckets[int(value)].append(index)
    ordered = np.array(sorted(buckets), dtype=np.uint64)
    found = set()

    def add(indices):
        support = tuple(int(positions[i]) for i in indices)
        affine_constant = int(np.bitwise_xor.reduce(signatures[list(indices)]) & np.uint64(1))
        found.add((support, affine_constant))

    for members in buckets.values():
        for pair in combinations(members, 2):
            add(pair)
    for i in range(len(positions)):
        pair_signatures = signatures[i] ^ signatures[i + 1:]
        targets = np.minimum(pair_signatures, pair_signatures ^ complement)
        where = np.searchsorted(ordered, targets)
        eligible = np.flatnonzero(where < len(ordered))
        eligible = eligible[ordered[where[eligible]] == targets[eligible]]
        for relative in eligible:
            j = i + 1 + int(relative)
            for k in buckets[int(targets[relative])]:
                if k > j:
                    add((i, j, k))
    return found, len(positions)


def registrations(rows, seen, limit, rng):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["calibration"] and seen[i].mean() >= .80:
            groups[row["group"]].append(i)
    selected = []
    keys = rng.permutation(sorted(groups))
    for group in keys:
        # PN disagreement and usable suffix coordinates both precede parity inspection.
        groups[group].sort(key=lambda i: (rows[i]["pn_errors"] / max(rows[i]["pn_seen"], 1)
                                          - .10 * seen[i].mean(), rng.random()))
    for repeat in range(2):
        selected.extend(groups[g][repeat] for g in keys if len(groups[g]) > repeat)
    return np.array(selected[:limit], dtype=int)


def candidate_score(bits, seen, groups, columns, affine, origins=None):
    x = bits[:, columns].copy()
    valid = seen[:, columns].copy()
    if origins is not None:
        x[:, -1] = x[origins, -1]
        valid[:, -1] = valid[origins, -1]
    available = valid.all(axis=1)
    match = (np.bitwise_xor.reduce(x, axis=1) == affine) & available
    _, inv = np.unique(groups, return_inverse=True)
    n = np.bincount(inv, weights=available)
    k = np.bincount(inv, weights=match)
    active = n > 0
    return {"known": int(available.sum()), "matches": int(match.sum()), "groups": int(active.sum()),
            "group_accuracy": float(np.mean(k[active] / n[active])) if active.any() else None}


def shifted_scores(bits, seen, groups, columns, affine, origins):
    first, last = columns[:-1], columns[-1]
    combined = np.bitwise_xor.reduce(bits[:, first], axis=1)
    available = seen[:, first].all(axis=1) & seen[origins, last]
    matches = ((combined ^ bits[origins, last]) == affine) & available
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    n = np.add.reduceat(available[:, order], starts, axis=1)
    k = np.add.reduceat(matches[:, order], starts, axis=1)
    return [float(np.mean(hit[known > 0] / known[known > 0])) if known.any() else None
            for hit, known in zip(k, n, strict=True)]


def audit_label(rows, suffix, visible, control_rows, control_bits, control_seen, label):
    fit = np.array([r["calibration"] for r in rows])
    confirm = ~fit
    groups = np.array([r["group"] for r in rows])
    indices = np.flatnonzero(fit)
    held_groups = groups[confirm]
    ordinary = np.array([not r["calibration"] for r in control_rows])
    ordinary_groups = np.array([r["group"] for r in control_rows])[ordinary]
    rng = np.random.default_rng(SEED)
    trial_records, nominated = [], set()
    registered_sets = {}
    for _ in range(TRIALS):
        chosen = registrations(rows, visible, 24 if label == 66 else 20, rng)
        key = tuple(sorted(chosen.tolist()))
        if key not in registered_sets:
            registered_sets[key] = exact_candidates(suffix[chosen], visible[chosen]) if len(chosen) >= 16 else (set(), 0)
        prior, usable = registered_sets[key]
        nominated.update(prior)
        trial_records.append({"row_indices": chosen.tolist(), "groups": len(np.unique(groups[chosen])),
                              "varying_fully_visible_columns": usable, "candidates": len(prior)})
    parity_fit = suffix[indices]
    seen_fit = visible[indices]
    groups_fit = groups[indices]
    held = suffix[confirm], visible[confirm], held_groups
    controls = np.array(list(control_origins(held_groups, np.random.default_rng(SEED))))
    calibration_pass = defaultdict(list)
    for columns, affine in sorted(nominated):
        cal = candidate_score(parity_fit, seen_fit, groups_fit, columns, affine)
        if cal["groups"] < MIN_GROUPS or cal["group_accuracy"] < FIT_MINIMUM:
            continue
        calibration_pass[len(columns)].append((columns, affine, cal))
    retained = []
    for passing in calibration_pass.values():
        passing.sort(key=lambda item: (-item[2]["group_accuracy"], -item[2]["groups"], -item[2]["known"], item[0]))
        retained.extend(passing[:FAMILY_LIMIT_PER_WEIGHT])
    validated = []
    for columns, affine, cal in retained:
        independent = candidate_score(*held, columns, affine)
        if independent["groups"] >= MIN_GROUPS:
            scores = shifted_scores(*held, columns, affine, controls)
        else:
            scores = []
        nonterminal = candidate_score(control_bits[ordinary], control_seen[ordinary], ordinary_groups, columns, affine)
        validated.append({"coordinates": list(columns), "constant": affine, "calibration": cal,
                          "held_padding": independent, "held_nonterminal": nonterminal, "group_shift_scores": scores})
    return {"registration_trials": trial_records, "unique_nominations": len(nominated),
            "calibration_pass_counts": {str(k): len(v) for k, v in sorted(calibration_pass.items())},
            "frozen_candidates": validated}


def cohort(archive, key, rows):
    edge, end = rows[0]["edge"], rows[0]["length"]
    return tuple(np.unpackbits(archive[key + suffix], axis=1)[:, edge:end].astype(bool)
                 for suffix in ("_bits", "_seen"))


def synthetic_control():
    rng = np.random.default_rng(SEED)
    values = rng.integers(0, 2, size=(64, 64), dtype=np.uint8)
    values[:, 17] = values[:, 5] ^ 1
    values[:, 37] = values[:, 13] ^ values[:, 24]
    seen = np.ones_like(values, dtype=bool)
    seen[:24, 60] = False
    expected = {((5, 17), 1), ((13, 24, 37), 0)}
    candidates, count = exact_candidates(values[:24], seen[:24])
    held = [candidate_score(values[24:], seen[24:], np.arange(40) // 4, idx, sign)
            for idx, sign in sorted(expected & candidates)]
    return {"nominations": sorted(candidates), "expected": sorted(expected), "columns": count,
            "held_injected_relations": held}


def family_diagnostic(formats):
    # Center each held result by its own grouped mismatch baseline, then correct for the search family.
    values, witnesses = [], []
    for label, row in formats.items():
        for candidate in row["frozen_candidates"]:
            shifts = candidate["group_shift_scores"]
            if len(shifts) != REPETITIONS or any(value is None for value in shifts) or np.std(shifts) < 1e-10:
                continue
            values.append((np.array([candidate["held_padding"]["group_accuracy"], *shifts])
                           - np.mean(shifts)) / np.std(shifts))
            witnesses.append({"label": label, "coordinates": candidate["coordinates"]})
    if not values:
        return {"usable_candidates": 0}
    matrix = np.array(values)
    maximum = matrix.max(axis=0)
    leader = witnesses[int(matrix[:, 0].argmax())]
    return {"usable_candidates": len(matrix), "leading": leader,
            "scores_equal_or_better_including_observed": int((maximum >= maximum[0]).sum()),
            "maxima_standardized": maximum.tolist()}


def main():
    require_run_context()
    inputs = {}
    info = read(checked_artifact(EXTRACTION, "E082", "gain_padded_words.json", inputs))["cohorts"]
    data = checked_artifact(EXTRACTION, "E082", "gain_padded_words.npz", inputs)
    formats = {}
    with np.load(data, allow_pickle=False) as z:
        for label in (66, 70):
            short, ordinary = f"short_{label}", f"nonterminal_{label}"
            formats[str(label)] = audit_label(info[short], *cohort(z, short, info[short]),
                                              info[ordinary], *cohort(z, ordinary, info[ordinary]), label)
            result = formats[str(label)]
            print(label, "nominations", result["unique_nominations"], "frozen", len(result["frozen_candidates"]), flush=True)
    report = {"schema": "starlink-sparse-padding-checks/v1", "seed": SEED, "formats": formats,
              "maximum_held_candidates_per_weight_and_format": FAMILY_LIMIT_PER_WEIGHT,
              "synthetic_control": synthetic_control(), "family_diagnostic": family_diagnostic(formats),
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E082")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sparse_padding_checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
