"""Fit a 512-bit cross-rate extension and eliminate the unobserved parity block."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH, checks
from audit_cross_rate_checks import endpoint_words, original_words, record, scores, shift_systematic
from audit_ut_repeated_data import checked_artifact, read
from recover_mcs66_full_code import SYSTEMATIC as ORIGINAL_SYSTEMATIC
from recover_mcs66_systematic_checks import LIFT, cyclic_response, packed_arrays, project_checks
from sparse_code5_core import binary_rank


SYSTEMATIC = ORIGINAL_SYSTEMATIC + LIFT
EXTENDED_BITS = BIT_LENGTH + LIFT
RELATIONS = ((0,), (2,), (4,), (5,), (7,), (1, 6), (3, 6))
MIN_GAIN = .15


def lift_family(rows):
    columns = np.where(rows < ORIGINAL_SYSTEMATIC, rows + LIFT,
                       np.where(rows < SYSTEMATIC, rows - ORIGINAL_SYSTEMATIC + BIT_LENGTH, rows))
    return np.sort(columns, axis=1)


def xor_families(left, right):
    return np.array([sorted(set(a) ^ set(b)) for a, b in zip(left, right, strict=True)])


def append_tap(rows, offset):
    if offset is None:
        return rows
    return np.column_stack(((offset + np.arange(LIFT)) % LIFT, rows))


def fit_first_block(support, bits, masks, groups):
    """Compare no tap against every one-tap orbit on the same seen coordinates."""
    value, seen = project_checks(bits, masks, support)
    early_seen = masks[:, :LIFT]
    works = (early_seen.mean(axis=1) > .97) & (seen.mean(axis=1) > .85)
    group_gains, group_scores = [], []
    for group in np.unique(groups[works]):
        take = works & (groups == group)
        left = early_seen[take]
        early_sign = np.where(left, 1 - 2 * bits[take, :LIFT].astype(float), 0)
        right = seen[take]
        parity_sign = np.where(right, 1 - 2 * value[take].astype(float), 0)
        overlap = cyclic_response(left, right).round().astype(int)
        if np.any(overlap == 0):
            raise ValueError("Calibration group has unmatched tap comparisons")
        with_tap = cyclic_response(early_sign, parity_sign).round() / overlap
        without_tap = cyclic_response(left, parity_sign).round() / overlap
        group_gains.append((with_tap - without_tap) / 2)
        group_scores.append((with_tap + 1) / 2)
    gains = np.mean(group_gains, axis=0)
    predicted = np.mean(group_scores, axis=0)
    ranking = np.argsort(-gains, kind="stable")[:5]
    tap = int(ranking[0]) if gains[ranking[0]] >= MIN_GAIN else None
    stats = [{"offset": int(k), "paired_group_accuracy_gain": float(gains[k]),
              "candidate_group_accuracy": float(predicted[k])} for k in ranking]
    return tap, {"accepted_first_block_offset": tap, "calibration_words": int(works.sum()),
                 "calibration_groups": np.unique(groups[works]).tolist(), "best_candidates": stats}


def projected_family(extended, relation):
    row = extended[relation[0]] if len(relation) == 1 else xor_families(*(extended[i] for i in relation))
    if np.any(row >= BIT_LENGTH):
        raise ValueError("Virtual parity did not cancel")
    return row


def dense_checks(families, length):
    matrix = np.zeros((len(families) * LIFT, length), dtype=np.uint8)
    for i, row in enumerate(families):
        matrix[i * LIFT + np.arange(LIFT)[:, None], row] = 1
    return matrix


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    with np.load(artifact("v126-mcs66-full-code", "E128", "mcs66_full_checks.npz"), allow_pickle=False) as z:
        source = z["row_support"].reshape(8, LIFT, 14)
    lifted = [lift_family(row) for row in source]
    expected = checks(artifact)[70]
    for old, aligned in zip(expected, (lifted[4], lifted[5]), strict=True):
        if not np.array_equal(old, aligned):
            raise ValueError("E120 positive-control families failed the cross-rate lift")

    run = "v125-cross-rate-checks"
    metadata = read(artifact(run, "E127", "cross_rate_checks.json"))
    excluded = metadata["archive_excluded_groups"]
    middle = {}
    with np.load(artifact(run, "E127", "cross_rate_middle_words.npz"), allow_pickle=False) as z:
        for code in (70, 88):
            if str(code) not in metadata["archive_middle_words"]:
                continue
            rows = metadata["archive_middle_words"][str(code)]
            bits, masks = packed_arrays(z, (f"mcs{code}_bits", f"mcs{code}_seen"))
            if code == 70:
                fit = np.array([r["calibration"] and r["group"] not in excluded for r in rows])
                calibration = bits[fit], masks[fit], np.array([r["group"] for r in rows])[fit]
            middle[code] = record(bits, masks, rows, "archive_middle", excluded)

    extended = lifted.copy()
    support, findings = [], []
    for relation in RELATIONS:
        row = projected_family(lifted, relation)
        offset, result = fit_first_block(row, *calibration)
        extended[relation[0]] = append_tap(lifted[relation[0]], offset)
        support.append(projected_family(extended, relation))
        findings.append({"source_families_zero_based": relation, **result,
                         "row_weight": support[-1].shape[1], "check_at_rotation_zero": support[-1][0].tolist()})
    for older, current in zip(expected, support[2:4], strict=True):
        if not np.array_equal(older, current):
            raise ValueError("A first-block correction changed the established positive controls")

    projected = dense_checks(support, BIT_LENGTH)
    larger = dense_checks(extended, EXTENDED_BITS)
    ranks = {"projected": binary_rank(projected), "projected_parity": binary_rank(projected[:, SYSTEMATIC:]),
             "extended": binary_rank(larger), "extended_parity": binary_rank(larger[:, SYSTEMATIC:])}
    if list(ranks.values()) != [3584, 3584, 4096, 4096]:
        raise ValueError(f"Unexpected channel-code ranks: {ranks}")
    run = "v120-cross-mcs-checks"
    endpoints = read(artifact(run, "E122", "cross_mcs_checks.json"))
    raw70 = original_words(artifact)[70]
    with np.load(artifact(run, "E122", "cross_mcs_observations.npz"), allow_pickle=False) as z:
        for code in (70, 88):
            cohorts = list(endpoint_words(z, endpoints, code, excluded))
            if code in middle:
                cohorts.append(middle[code])
            if code == 70:
                cohorts.append(raw70)
            for row, result in zip(support, findings, strict=True):
                compared = scores(row, shift_systematic(row, SYSTEMATIC), cohorts)
                check = compared["archive_nonterminal"]
                if check["reference"]["group_accuracy"] - check["shifted_systematic"]["group_accuracy"] < .04:
                    raise ValueError(f"Source families {result['source_families_zero_based']} failed MCS {code} holdout")
                result[f"mcs{code}_validation"] = compared

    output = experiment_output("E131")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "mcs70_full_checks.npz", projected_matrix=np.packbits(projected, axis=1),
                        extended_matrix=np.packbits(larger, axis=1),
                        **{f"extended_family_{i}": rows for i, rows in enumerate(extended)})
    receipt = {"schema": "starlink-mcs70-full-code-recovery/v1", "systematic_bits": SYSTEMATIC,
               "codeword_bits": BIT_LENGTH, "virtual_parity_interval": [BIT_LENGTH, EXTENDED_BITS],
               "minimum_paired_calibration_gain": MIN_GAIN, "archive_overlapping_groups_excluded": excluded,
               "ranks": ranks, "relations": findings,
               "artifact_input_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    (output / "mcs70_full_code_recovery.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(ranks, [(r['source_families_zero_based'], r['accepted_first_block_offset']) for r in findings], flush=True)


if __name__ == "__main__":
    main()
