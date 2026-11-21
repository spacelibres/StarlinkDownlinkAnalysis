"""Search MCS 66 parity projections while retaining the active short prefix."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_sparse_padding_checks import candidate_score, exact_candidates, registrations
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, LIFT, packed_arrays


PREFIX = 1024
SEED = 2026112184
PAIR_ID_BITS = 13
PAIR_BITS = 2 * PAIR_ID_BITS
ID_MASK = (1 << PAIR_ID_BITS) - 1
ANCHOR_OFFSETS = (0, 17, 63, 128, 257, 383, 460)


def signed_pair_index(signatures, subset, complement):
    n = len(subset)
    pairs = np.empty(n * (n - 1) // 2, dtype=np.uint64)
    ids = subset.astype(np.uint64)
    start = 0
    for i, left in enumerate(ids[:-1]):
        right = ids[i + 1:]
        value = signatures[left] ^ signatures[right]
        key = np.minimum(value, value ^ complement)
        pairs[start:start + len(right)] = (key << PAIR_BITS) | (left << PAIR_ID_BITS) | right
        start += len(right)
    pairs.sort()
    return pairs, pairs >> PAIR_BITS


def matching_pairs(pairs, keys, targets, complement):
    query = np.minimum(targets, targets ^ complement)
    lo = np.searchsorted(keys, query, side="left")
    hi = np.searchsorted(keys, query, side="right")
    for index in np.flatnonzero(hi > lo):
        for value in pairs[lo[index]:hi[index]]:
            item = int(value)
            yield int(index), (item >> PAIR_ID_BITS) & ID_MASK, item & ID_MASK


def mask_for_support(support):
    vector = 0
    for k in support:
        vector ^= 1 << int(k)
    return vector


def reduce_support(support, basis):
    vector = mask_for_support(support)
    while vector:
        pivot = vector.bit_length() - 1
        if pivot not in basis:
            break
        vector ^= basis[pivot]
    return vector


def insert_support(support, basis):
    vector = reduce_support(support, basis)
    if not vector:
        return False
    basis[vector.bit_length() - 1] = vector
    return True


def signatures_for(bits, mask, min_weight):
    if not 2 <= len(bits) <= 38 or bits.shape[1] > 1 << PAIR_ID_BITS:
        raise ValueError("Too many registration rows or columns for the packed index")
    ones = bits.sum(axis=0)
    coords = np.flatnonzero(mask.all(axis=0) & (ones >= min_weight) & (ones <= len(bits) - min_weight))
    signatures = bits[:, coords].astype(np.uint64).T @ (np.uint64(1) << np.arange(len(bits), dtype=np.uint64))
    return coords, signatures, np.uint64((1 << len(bits)) - 1)


def anchored_projections(bits, mask, weight):
    coords, signatures, complement = signatures_for(bits, mask, 1 if weight == 4 else 2)
    parity = np.flatnonzero(coords >= PREFIX)
    anchors = parity[np.isin((coords[parity] - PREFIX) % LIFT, ANCHOR_OFFSETS)]
    nominated = set()

    def add(indices):
        if len(set(indices)) != weight:
            return
        observed = tuple(sorted(int(coords[i]) for i in indices))
        sign = int(np.bitwise_xor.reduce(signatures[list(indices)]) & np.uint64(1))
        nominated.add((observed, sign))

    if weight == 4:
        pairs, keys = signed_pair_index(signatures, np.arange(len(coords)), complement)
        for i in anchors:
            for j, k, h in matching_pairs(pairs, keys, signatures[i] ^ signatures, complement):
                add((i, j, k, h))
    else:
        pairs, keys = signed_pair_index(signatures, parity, complement)
        first = np.flatnonzero(coords < LIFT)
        second = np.flatnonzero((coords >= LIFT) & (coords < PREFIX))
        mixed = (signatures[first, None] ^ signatures[second][None, :]).ravel()
        ones = np.bitwise_count(mixed)
        support = np.flatnonzero((ones >= 3) & (ones <= len(bits) - 3))
        for i in anchors:
            for pair_index, k, h in matching_pairs(pairs, keys, signatures[i] ^ mixed[support], complement):
                left, right = divmod(int(support[pair_index]), len(second))
                add((i, first[left], second[right], k, h))
    return nominated, {"eligible_columns": len(coords), "anchors": coords[anchors].tolist()}


def longer_registration(rows, visible, count, rng):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        if row["calibration"] and visible[i].mean() >= .90:
            groups[row["group"]].append(i)
    for members in groups.values():
        members.sort(key=lambda i: (rows[i]["pn_errors"] / max(rows[i]["pn_seen"], 1) - .1 * visible[i].mean(), i))
    if not groups:
        return np.array([], dtype=int)
    permuted = rng.permutation(sorted(groups))
    take = [groups[g][j] for j in range(max(map(len, groups.values()))) for g in permuted if j < len(groups[g])]
    return np.array(take[:count], dtype=int)


def normalized_shape(support):
    par = [(divmod(x - PREFIX, LIFT)) for x in support if x >= PREFIX]
    sys = [(divmod(x, LIFT)) for x in support if x < PREFIX]
    first_block = min(b for b, _ in par)
    return min((tuple(sorted((b, (o - phase) % LIFT) for b, o in par)),
                tuple(sorted((b, (o - phase) % LIFT) for b, o in sys)))
               for block, phase in par if block == first_block)


def audit(rows, bits, mask, pn, edge, prior_support):
    bound = np.array([r["bound"] for r in rows])
    fit = np.array([r["calibration"] for r in rows])
    short = bound <= PREFIX
    rows = [row for row, include in zip(rows, short, strict=True) if include]
    before_pn = np.arange(PREFIX) < bound[short, None]
    bits, mask = bits[short], mask[short]
    fit = fit[short]
    group = np.array([r["group"] for r in rows])
    values = np.column_stack(((bits[:, :PREFIX] ^ pn[:PREFIX]) & before_pn, bits[:, edge:]))
    seen = np.column_stack((mask[:, :PREFIX] | ~before_pn, mask[:, edge:]))
    parity_seen = mask[:, edge:]
    rng = np.random.default_rng(SEED)
    standard = registrations(rows, parity_seen, 30, rng)
    longer = longer_registration(rows, parity_seen, 36, rng)
    picked = {"two_and_three": standard, "four": standard, "five": longer}
    candidates, trace = set(), []
    for purpose, indices in picked.items():
        if purpose == "two_and_three":
            new, count = exact_candidates(values[indices], seen[indices])
            details = {"eligible_columns": count}
        else:
            new, details = anchored_projections(values[indices], seen[indices], 4 if purpose == "four" else 5)
        candidates.update(new)
        trace.append({"purpose": purpose, "picked_frames": [rows[i]["frame"] for i in indices],
                      "groups": len(np.unique(group[indices])), "nominations": len(new), **details})
        print(purpose, "nominated", len(new), flush=True)
    known = {}
    for row in prior_support:
        insert_support(row[row >= edge] - edge, known)
    models = defaultdict(list)
    fits = values[fit], seen[fit], group[fit]
    dependent = 0
    for support, sign in sorted(candidates):
        if sum(coordinate >= PREFIX for coordinate in support) < 2:
            continue
        training = candidate_score(*fits, support, sign)
        if training["groups"] < 8 or training["group_accuracy"] < .90:
            continue
        if not reduce_support([k - PREFIX for k in support if k >= PREFIX], known):
            dependent += 1
            continue
        models[normalized_shape(support)].append({"coordinates": support, "fixed_affine": sign, "training": training})
    ranked = sorted(models, key=lambda k: (-len(models[k]), k))
    survivors = [{"parity_shape": par, "early_shape": sys, "matching_rotations": len(models[par, sys]),
                  "support": models[par, sys]} for par, sys in ranked if len(models[par, sys]) >= 3]
    return {"registrations": trace, "distinct_nominations": len(candidates), "prior_parity_rank": len(known),
            "dependent_calibration_pass": dependent, "independent_shapes_before_rotation_screen": len(models),
            "retained": survivors}


def main():
    require_run_context()
    inputs = {}
    info = read(checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.json", inputs))
    data = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    support_path = checked_artifact("v081-mcs66-systematic-checks", "E083", "mcs66_partial_checks.npz", inputs)
    with np.load(data, allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ("bits", "mask"))
    with np.load(support_path, allow_pickle=False) as z:
        support = z["row_support"]
    edge = info["systematic_end"]
    pn = equality_predictions(np.arange(edge), pn_bits(), info["pn_model"]).astype(bool)
    result = audit(info["rows"], bits, mask, pn, edge, support)
    result.update(schema="starlink-mcs66-early-projections/v1", seed=SEED,
                  inputs_sha256={p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()})
    out = experiment_output("E084")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mcs66_early_projections.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("retained shapes", len(result["retained"]), "known-span matches", result["dependent_calibration_pass"])


if __name__ == "__main__":
    main()
