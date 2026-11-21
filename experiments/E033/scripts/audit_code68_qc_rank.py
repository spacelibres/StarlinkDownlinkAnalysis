"""Audit identifiability and exact contradictions in Code 68 cyclic models."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import binary_blocks, evaluate, pack_blocks
from quotient_lift_core import pascal_transform
from receiver_models import pn_bits

K, N = 6144, 8192
LIFTS = (256, 512, 1024, 2048)
SEED = 2026112133
ALT_MARGIN = .02
SCREEN_DEPTH = 32


def truncated_product(a, b, depth):
    if a.bit_count() < b.bit_count():
        a, b = b, a
    value = 0
    while b:
        term = b & -b
        value ^= a << (term.bit_length() - 1)
        b ^= term
    return value & ((1 << depth) - 1)


def valuation(value, zero_default):
    return (value & -value).bit_length() - 1 if value else zero_default


def unit_inverse(unit, depth):
    inverse = 1
    precision = 1
    while precision < depth:
        precision = min(2 * precision, depth)
        squared = truncated_product(inverse, inverse, precision)
        inverse = truncated_product(unit, squared, precision)
    return inverse


def diagonalize(train_u, train_v, hold_u, depth):
    """Carry row operations through the RHS and column operations through holdout rows."""
    a = [row.copy() for row in train_u]
    b = [row.copy() for row in train_v]
    hold = [row.copy() for row in hold_u]
    n, k = len(a), len(a[0])
    pivots = []
    for i in range(min(n, k)):
        d, r, c = min((valuation(a[r][c], depth), r, c)
                      for r in range(i, n) for c in range(i, k))
        if d == depth:
            break
        a[i], a[r] = a[r], a[i]
        b[i], b[r] = b[r], b[i]
        for row in [*a, *hold]:
            row[i], row[c] = row[c], row[i]
        inverse = unit_inverse(a[i][i] >> d, depth - d)
        a[i] = [truncated_product(x, inverse, depth) for x in a[i]]
        b[i] = [truncated_product(x, inverse, depth) for x in b[i]]
        for r in range(i + 1, n):
            factor = a[r][i] >> d
            if factor:
                a[r] = [x ^ truncated_product(y, factor, depth) for x, y in zip(a[r], a[i], strict=True)]
                b[r] = [x ^ truncated_product(y, factor, depth) for x, y in zip(b[r], b[i], strict=True)]
        for c in range(i + 1, k):
            factor = a[i][c] >> d
            if factor:
                for row in [*a, *hold]:
                    row[c] ^= truncated_product(row[i], factor, depth)
        pivots.append(d)
    return b, hold, pivots


def solve_and_predict(train_u, train_v, hold_u, hold_v, depth):
    b, hold, pivots = diagonalize(train_u, train_v, hold_u, depth)
    k, p = len(train_u[0]), len(train_v[0])
    row_degrees = pivots + [depth] * (len(b) - len(pivots))
    contradiction = min((valuation(x & ((1 << d) - 1), depth)
                         for row, d in zip(b, row_degrees, strict=True) for x in row), default=depth)
    rank = sum(depth - v for v in pivots)
    result = {"rank": rank, "generator_kernel_bits_per_column": k * depth - rank,
              "pivot_valuations": pivots, "first_inconsistent_degree": contradiction if contradiction < depth else None}

    # Restrict to the largest exact prefix if the higher-degree observation is inconsistent.
    valid_depth = contradiction
    free_orders = [max(0, valid_depth - v) for v in pivots] + [0] * (k - len(pivots))
    parameters = [[(x % (1 << valid_depth)) >> d for x in row]
                  for row, d in zip(b[:len(pivots)], pivots, strict=True)] + [[0] * p for _ in range(k - len(pivots))]
    fixed = []
    errors = []
    raw_errors = []
    for coeffs, observed in zip(hold, hold_v, strict=True):
        prefix = min([valid_depth, *(r + valuation(c, valid_depth) for r, c in zip(free_orders, coeffs, strict=True))])
        predicted = [0] * p
        for coefficient, target in zip(coeffs, parameters, strict=True):
            for j, v in enumerate(target):
                predicted[j] ^= truncated_product(coefficient, v, prefix)
        mask = (1 << prefix) - 1
        residual = [(a ^ b) & mask for a, b in zip(predicted, observed, strict=True)]
        fixed.append(prefix)
        errors.append(sum(x.bit_count() for x in residual))
        if prefix == depth:
            raw_errors.append(int(pascal_transform(binary_blocks([residual], depth).reshape(p, depth)).sum()))
        else:
            raw_errors.append(None)
    result["prediction_fixed_prefixes"] = fixed
    result["held_mismatch_coefficients"] = errors
    result["held_full_parity_mismatches"] = raw_errors
    return result


def project_words(words, lift, depth):
    nblocks = words.shape[1] // lift
    blocks = words.reshape(len(words), nblocks, lift)
    if depth < lift:
        if depth & (depth - 1):
            raise ValueError("Projection length must be a power of two")
        blocks = np.bitwise_xor.reduce(blocks.reshape(len(words), nblocks, -1, depth), axis=2)
    values = pascal_transform(blocks)
    return pack_blocks(values.reshape(len(words), -1), depth)


def assess(words, training, lift, depth=None, affine=False):
    q = lift if depth is None else depth
    u = project_words(words[:, :K], lift, q)
    v = project_words(words[:, K:], lift, q)
    if affine:
        for row in u:
            row.append(1)
    t, h = np.flatnonzero(training), np.flatnonzero(~training)
    pick = lambda values, indices: [values[i] for i in indices]
    return solve_and_predict(pick(u, t), pick(v, t), pick(u, h), pick(v, h), q)


def reference_repairs(bits, expected, words):
    adjusted = bits.copy()
    repairs = []
    mask = np.zeros(bits.shape, dtype=bool)
    for i, word in enumerate(words):
        interval = word["fixed_zero_source"]
        if interval["accepted_144"]:
            lo = interval["start"]
            mask[i, lo:K] = True
            for pos in np.flatnonzero(bits[i, lo:K] != expected[lo:K]) + lo:
                repairs.append({"word": i, "bit": int(pos)})
            adjusted[i, lo:K] = expected[lo:K]
    return adjusted, mask, repairs


def second_points(symbols, labels):
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    points = (levels[:, None] + 1j * levels[None, :]).ravel()
    distance = abs(symbols[..., None] - points) ** 2
    ids = np.argsort(distance, axis=-1)[..., :2]
    nearest = labels[ids[..., 0]]
    alternative = labels[ids[..., 1]]
    costs = np.take_along_axis(distance, ids, axis=-1)
    return nearest, alternative, costs[..., 1] - costs[..., 0]


def repair_screen(words, confidence_mask, nearest, alternative, margins, training, lift):
    options = []
    best = None
    for w, sym in zip(*np.where((margins < ALT_MARGIN) & training[:, None]), strict=True):
        change = slice(sym * 4, sym * 4 + 4)
        flips = nearest[w, sym] != alternative[w, sym]
        if np.any(confidence_mask[w, change] & flips):
            continue
        options.append((int(w), int(sym), float(margins[w, sym])))
    with_repair = words.copy()
    for w, sym, cost in [(-1, 0, 0.), *options]:
        change = slice(sym * 4, sym * 4 + 4)
        if w >= 0:
            with_repair[w, change] ^= nearest[w, sym] ^ alternative[w, sym]
        approximate = assess(with_repair, training, lift, SCREEN_DEPTH)
        degree = approximate["first_inconsistent_degree"]
        merit = (SCREEN_DEPTH if degree is None else degree, -cost)
        if best is None or merit > best[0]:
            best = merit, w, sym, cost
        if w >= 0:
            with_repair[w, change] = words[w, change]
    merit, w, sym, cost = best
    if w >= 0:
        change = slice(sym * 4, sym * 4 + 4)
        with_repair[w, change] ^= nearest[w, sym] ^ alternative[w, sym]
    return {"options_count": len(options) + 1, "selected_word": w if w >= 0 else None,
            "selected_symbol": sym if w >= 0 else None, "selection_cost": cost, "screen_prefix": merit[0],
            "result": assess(with_repair, training, lift)}


def control(frames):
    rng = np.random.default_rng(SEED)
    output = {}
    inputs = rng.integers(0, 2, (len(frames), K), dtype=np.uint8)
    for z in LIFTS:
        p = rng.integers(0, 2, (K//z, N-K), dtype=np.uint8)
        enc = binary_blocks(evaluate(pack_blocks(inputs, z), pack_blocks(p, z), z), z)
        trials = {}
        for error_count in (0, 1, 4):
            word = np.hstack((inputs, enc))
            errors = rng.choice(word.size, size=error_count, replace=False)
            word.ravel()[errors] ^= 1
            trials[str(error_count)] = {"positions": errors.tolist(),
                "groups": {str(f): assess(word, frames != f, z) for f in np.unique(frames)}}
        output[str(z)] = trials
    return output


def main():
    require_run_context()
    sources = {}
    ref = checked_artifact("v030-legacy-pn-transfer-final", "E032", "legacy_pn_transfer.json", sources)
    data = checked_artifact("v021-code68-erasure-audit", "E023", "code68_observations.npz", sources)
    record = read(ref)
    descriptors = record["formats"]["68"]["words"]
    frames = np.array([w["frame"] for w in descriptors])
    pn = equality_predictions(np.arange(N), pn_bits(), record["pn_model"])
    with np.load(data, allow_pickle=False) as z:
        measured = z["gray_bits"]
        near, alternate, margin = second_points(z["symbols"], z["frozen_gray_mapping"])
    if not np.array_equal(near.reshape(-1, N), measured):
        raise ValueError("Archived hard labels disagree with the saved I/Q observations")
    assisted, mask, changes = reference_repairs(measured, pn, descriptors)
    for row in changes:
        row["two_nearest_distance_gap"] = float(margin[row["word"], row["bit"] // 4])
    representations = {"observed": measured, "reference_assisted": assisted,
                      "input_dewhitened": assisted ^ np.r_[pn[:K], np.zeros(N-K, dtype=np.uint8)],
                      "all_dewhitened": assisted ^ pn, "byte_reversed": assisted[:, np.arange(N) ^ 7]}
    model_tests = {}
    for name, words in representations.items():
        model_tests[name] = {str(z): {str(f): assess(words, frames != f, z) for f in np.unique(frames)} for z in LIFTS}
    affine_models = {
        name: {str(z): {str(f): assess(words, frames != f, z, affine=True) for f in np.unique(frames)} for z in LIFTS}
        for name, words in (("observed", measured), ("reference_assisted", assisted))
    }
    pooled = {str(z): assess(assisted, np.ones(len(frames), dtype=bool), z) for z in LIFTS}
    screen = {str(z): {str(f): repair_screen(assisted, mask, near, alternate, margin, frames != f, z)
                       for f in np.unique(frames)} for z in (1024, 2048)}
    output = experiment_output("E033")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "starlink-code68-qc-rank/v1", "inputs_sha256": {p.relative_to(ROOT).as_posix(): s for p, s in sources.items()},
        "K": K, "N": N, "frames_by_word": frames.tolist(), "seed": SEED,
        "reference_substitutions": changes, "model_tests": model_tests,
        "affine_model_tests": affine_models, "pooled_reference_assisted": pooled,
        "single_symbol_training_screen": screen, "max_second_point_cost": ALT_MARGIN, "screen_depth": SCREEN_DEPTH,
        "synthetic": control(frames),
        "scope": "Exact binary systematic QC model in specified label orders, and finite calibration-only alternate decisions. "
                 "Fixed coefficient counts are in t=x+1 and are not raw parity bit BER.",
    }
    (output / "code68_qc_rank.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Reference-assisted changes:", len(changes), "; full QC fits and guarded projections saved.", flush=True)


if __name__ == "__main__":
    main()
