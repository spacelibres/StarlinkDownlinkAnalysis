"""Score circulant-generator projections while retaining unsupported bit positions."""

from collections import defaultdict
from itertools import permutations, product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_qin_mcs6_structure import GUARD, K
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


N = 16384
SEED = 2026112139
LOG_GOOD = np.log(1.8)
LOG_BAD = np.log(.2)


def prepare(symbols, descriptors, transfer, pn):
    raw = (symbols.real < 0).astype(np.uint8)
    raw_erased = abs(symbols.real) <= .5*np.median(abs(symbols), axis=1, keepdims=True)
    shifted = raw.copy()
    shifted_erased = raw_erased.copy()
    for i, case in enumerate(transfer):
        carrier = case["carrier_transfer"]
        positions = carrier["coordinates"]
        fit = carrier["exact_erasure"]
        shifted_erased[i, positions] = True
        if "estimated" in fit:
            values = np.asarray(fit["estimated"])
            available = values >= 0
            locations = np.asarray(positions)[available]
            shifted[i, locations] = values[available]
            shifted_erased[i, locations] = False
    assisted = shifted.copy()
    assisted_erased = shifted_erased.copy()
    for i, row in enumerate(descriptors):
        run = row["fixed"]
        if run["accepted"]:
            window = slice(run["start"]+GUARD, min(K, run["stop"])-GUARD)
            assisted[i, window] = pn[window]
            assisted_erased[i, window] = False
    return {"raw": (raw, raw_erased), "edge": (shifted, shifted_erased),
            "edge_and_known_pn": (assisted, assisted_erased)}


def folds(bits, erasures, lift):
    blocks = bits.reshape(len(bits), -1, lift)
    unavailable = erasures.reshape(len(bits), -1, lift)
    even = slice(0, None, 2)
    odd = slice(1, None, 2)
    collapse = lambda selector: np.bitwise_xor.reduce(blocks[:, :, selector], axis=2)
    return (collapse(even), collapse(odd),
            unavailable[:, :, even].any(axis=2), unavailable[:, :, odd].any(axis=2))


def comparisons(arrays, lift, first_masks):
    even, odd, masked_even, masked_odd = arrays
    k = K//lift
    masks = ((np.arange(2**k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    for j in range(k, N//lift):
        if first_masks is None:
            odd_masks = masks
            expected = even[:, j] ^ odd[:, j]
            reference_available = ~(masked_even[:, j] | masked_odd[:, j])
        else:
            selected = first_masks[j-k]
            if selected is None:
                yield None
                continue
            odd_masks = masks ^ masks[selected]
            expected = odd[:, j]
            reference_available = ~masked_odd[:, j]
        pred = ((masks @ even[:, :k].T) ^ (odd_masks @ odd[:, :k].T)) & 1
        available = ((masks @ masked_even[:, :k].T) + (odd_masks @ masked_odd[:, :k].T)) == 0
        yield pred, available, expected, reference_available


def held_orders(training, groups):
    by_size = defaultdict(list)
    for group in np.unique(groups[~training]):
        indices = np.flatnonzero((groups == group) & ~training)
        by_size[len(indices)].append(indices)
    bins = list(by_size.values())
    for order in product(*(permutations(blocks) for blocks in bins)):
        mapping = np.arange(len(training))
        for original, reassign in zip(bins, order, strict=True):
            for left, right in zip(original, reassign, strict=True):
                mapping[left] = right
        yield mapping


def fit_fold(data, lift, training, groups, first_masks=None):
    held = np.flatnonzero(~training)
    fitting = np.flatnonzero(training)
    orders = list(held_orders(training, groups))
    result = []
    null_scores = np.zeros(len(orders))
    for comparison in comparisons(data, lift, first_masks):
        if comparison is None:
            result.append({"selected_mask": None, "status": "no_first_coefficient"})
            continue
        pred, inputs_valid, y, y_valid = comparison
        valid = inputs_valid & y_valid
        correct = pred == y
        n = valid[:, fitting].sum(axis=1)
        errors = (valid[:, fitting] & ~correct[:, fitting]).sum(axis=1)
        group_count = sum(valid[:, training & (groups == g)].any(axis=1).astype(int) for g in np.unique(groups[training]))
        eligible = np.flatnonzero((n >= K//lift+2) & (group_count >= 2))
        if not len(eligible):
            result.append({"selected_mask": None, "status": "insufficient_coverage"})
            continue
        score = (n-errors)*LOG_GOOD + errors*LOG_BAD
        best = max(eligible, key=lambda m: (round(score[m], 10), group_count[m], -m))
        for i, mapping in enumerate(orders):
            usable = inputs_valid[best, held] & y_valid[mapping[held]]
            success = (pred[best, held] == y[mapping[held]]) & usable
            null_scores[i] += success.sum()*LOG_GOOD + (usable.sum()-success.sum())*LOG_BAD
        result.append({"selected_mask": int(best), "candidate_count": int(len(n)), "eligible_count": len(eligible),
                       "calibration_usable": int(n[best]), "calibration_errors": int(errors[best]),
                       "calibration_groups": int(group_count[best]), "selected_score": float(score[best]),
                       "calibration_zero_error_candidates": int(np.count_nonzero(errors[eligible] == 0)),
                       "held_reference": y[held].tolist(), "held_predicted": pred[best, held].tolist(),
                       "held_input_valid": inputs_valid[best, held].tolist(), "held_valid": valid[best, held].tolist(),
                       "held_correct": correct[best, held].tolist()})
    identity = next(i for i, mapping in enumerate(orders) if np.array_equal(mapping, np.arange(len(training))))
    observed = null_scores[identity]
    return {"results": result, "fit_words": fitting.tolist(), "held_words": held.tolist(),
            "observed_score": float(observed), "permutation_scores": null_scores.tolist(),
            "permutations_not_below_observed": int(np.count_nonzero(null_scores >= observed-1e-10))}


def fit_pair(bits, erasures, groups, train, lift):
    data = folds(bits, erasures, lift)
    a = fit_fold(data, lift, train, groups)
    result = {"coefficient_0": a}
    if lift == 1024:
        masks = [r["selected_mask"] for r in a["results"]]
        result["coefficient_1"] = fit_fold(data, lift, train, groups, masks)
    return result


def synthetic_control():
    rng = np.random.default_rng(SEED)
    lift = 1024
    rows = 64
    input_blocks = rng.integers(0, 2, (rows, 10, lift), dtype=np.uint8)
    parity_blocks = np.zeros((rows, 6, lift), dtype=np.uint8)
    m0 = np.zeros(6, dtype=int)
    m1 = np.zeros(6, dtype=int)
    for i, j in product(range(10), range(6)):
        shifts = rng.integers(0, lift, rng.integers(1, 4))
        for shift in shifts:
            parity_blocks[:, j] ^= np.roll(input_blocks[:, i], shift, axis=-1)
        m0[j] |= (len(shifts) % 2) << i
        m1[j] |= (sum(shifts) % 2) << i
    words = np.concatenate((input_blocks, parity_blocks), axis=1).reshape(rows, -1)
    erasures = np.zeros(words.shape, dtype=bool)
    erasures[np.arange(0, rows, 4), rng.integers(0, N, rows//4)] = True
    group = np.arange(rows)//8
    models = fit_pair(words, erasures, group, group < 6, lift)
    for expected, key in ((m0, "coefficient_0"), (m1, "coefficient_1")):
        found = [item["selected_mask"] for item in models[key]["results"]]
        if not np.array_equal(expected, found):
            raise ValueError("Synthetic fold failed to recover known masks")
    return {"expected_coefficient_0": m0.tolist(), "expected_coefficient_1": m1.tolist(), "fits": models}


def main():
    require_run_context()
    inputs = {}
    audit = read(checked_artifact("v034-qin-low-mcs-audit", "E036", "qin_low_mcs_audit.json", inputs))
    symbols = checked_artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_symbols.npz", inputs)
    transfer = read(checked_artifact("v036-qin-mcs6-transfer", "E038", "qin_mcs6_transfer.json", inputs))
    descriptors = audit["formats"]["6"]["words"]
    training = np.array([r["calibration"] for r in descriptors])
    groups = np.array([r["group"] for r in descriptors])
    pn = equality_predictions(np.arange(N), pn_bits(), audit["pn_model"])
    with np.load(symbols, allow_pickle=False) as data:
        branches = prepare(data["mcs6_symbols"], descriptors, transfer["words"], pn)
    fits = {}
    for name, (b, erased) in branches.items():
        fits[name] = {"erasures_by_word": erased.sum(axis=1).tolist(),
                      "lifts": {lift: fit_pair(b, erased, groups, training, lift) for lift in (1024, 2048)}}
    result = {"schema": "starlink-qin-mcs6-folded/v1", "K": K, "N": N,
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "group_seed": audit["whole_group_seed"], "groups_by_word": groups.tolist(),
              "training_by_word": training.tolist(), "tested_reference_match_probability": .9,
              "synthetic_seed": SEED, "measured": fits, "synthetic": synthetic_control()}
    out = experiment_output("E039")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qin_mcs6_folded.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Audited all masks at the declared lifts on", len(descriptors), "measured words", flush=True)


if __name__ == "__main__":
    main()
