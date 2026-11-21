"""Test an externally frozen 60-bit tail template on narrowband projections."""

from functools import lru_cache
import hashlib
import json
from math import comb
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_h2_projection import SELECTION_ROOT, TRACK, WAV_MODEL, band_masks
from layout_core import NONPILOT
from receiver_models import pn_reference

NPZ_ROOT = ROOT / "runs/v006-guard06-variance/artifacts/experiments/E008/results"
NPZ = NPZ_ROOT / "npz_crossvalidated_observations.npz"
RECEIVER = NPZ_ROOT / "npz_information_audit.json"
STRONG_GUARD = ROOT / "runs/v006-guard15-variance/artifacts/experiments/E008/results"
GUARD_NPZ = STRONG_GUARD / NPZ.name
GUARD_RECEIVER = STRONG_GUARD / RECEIVER.name
WIDEBAND = ROOT / "runs/v023-binary-tail-final/artifacts/experiments/E025/results/ut_binary_tail.json"
WIDEBAND_IQ = WIDEBAND.with_suffix(".npz")
QIN = ROOT / "references/qin/referenceTemplate.mat"
PN_WRONG_SHIFT = 1
SEED = 2026112126
CONTROLS = 399
PERIOD = 60
NRESOURCE = len(NONPILOT)
NFRAME = 300 * NRESOURCE


def ambiguity(template, positions):
    """Count exact aliases under noiseless phases and fixed modulo-four weights."""
    visited = np.unique(positions % len(template))
    alternatives = 1
    for lane in range(4):
        observed = visited[visited % 4 == lane]
        remaining = len(template[lane::4]) - len(observed)
        ones = int(np.count_nonzero(template[lane::4]) - np.count_nonzero(template[observed]))
        alternatives *= comb(remaining, ones)
    return {
        "ideal_symbol_count": len(positions),
        "distinct_template_positions": len(visited),
        "same_lane_composition_compatible_templates": alternatives,
    }


def bank(template, indices, masks=None):
    """One unit contrast vector per phase and carrier, with carrier DC removed."""
    if masks is None:
        masks = np.ones((1, *indices.shape), dtype=bool)
    phases = np.arange(len(template))[:, None, None]
    wave = template[(indices[None, :, :] + phases) % len(template)]
    waves = wave[None, :, :, :] * masks[:, None, :, :]
    count = masks.sum(axis=1)
    mean = waves.sum(axis=2) / np.maximum(count[:, None, :], 1)
    contrast = (waves - mean[:, :, None, :]) * masks[:, None, :, :]
    contrast *= (count >= 3)[:, None, None, :]
    norm = np.linalg.norm(contrast, axis=2, keepdims=True)
    unit = contrast / np.where(norm > 1e-12, norm, 1)
    return unit.reshape(-1, *indices.shape)


def center(values):
    return values - values.mean(axis=1, keepdims=True)


def covariance(observations):
    residual = center(observations)
    matrix = np.einsum("fsc,ftc->cst", residual, residual.conj()).real
    energy = float(np.sum(abs(residual) ** 2))
    return matrix / energy if energy else matrix


def apply_covariance(vectors, matrix):
    return np.einsum("hsc,cst,htc->h", vectors, matrix, vectors, optimize=True)


@lru_cache(maxsize=5)
def masks_for(indices_key, starts):
    indices = np.array(indices_key)
    all_masks = indices[None, :, :] >= np.array(starts)[:, None, None]
    # Keep equivalent supports only once. All cyclic phases are searched later.
    _, unique = np.unique(all_masks.reshape(len(starts), -1), axis=0, return_index=True)
    use = [i for i in sorted(unique) if np.any(all_masks[i].sum(axis=0) >= 3)]
    return all_masks[use], [starts[i] for i in use]


def template_test(template, coordinates, masks, left, right, projectors=None):
    vectors = bank(template, coordinates, masks)
    eligible = np.any(vectors != 0, axis=(1, 2))
    if projectors is None:
        scores = [apply_covariance(vectors, m) for m in (left, right)]
    else:
        scores = []
        for matrix, projector in zip((left, right), projectors, strict=True):
            transformed = np.einsum("cst,htc->hsc", projector, vectors)
            norms = np.linalg.norm(transformed, axis=1, keepdims=True)
            transformed = np.divide(transformed, norms, out=np.zeros_like(transformed), where=norms > 1e-10)
            scores.append(apply_covariance(transformed, matrix))
    if not np.any(eligible):
        return {"held_score": 0., "selection": [], "nonconstant_candidates": 0}
    choices = [int(np.argmax(np.where(eligible, data, -np.inf))) for data in scores]
    held = [float(scores[1 - fold][j]) for fold, j in enumerate(choices)]
    return {
        "held_score": float(np.mean(held)),
        "selection": [
            {"support_index": j // PERIOD, "phase_in_absolute_resource_order": j % PERIOD,
             "training_score": float(scores[fold][j]), "held_score": held[fold]}
            for fold, j in enumerate(choices)
        ],
        "nonconstant_candidates": int(eligible.sum()),
    }


def short_slots(q):
    slots = np.arange(297, 302)
    return (slots - 2)[:, None] * NRESOURCE + q[None, :]


def code_indices(carriers):
    signed = np.asarray(carriers) - 1024
    return np.array([int(np.flatnonzero(NONPILOT == k)[0]) for k in signed])


def run_contexts(cube, carrier_ids, frame_groups, train, reserved, interval):
    q = code_indices(carrier_ids)
    short = short_slots(q)
    lower, upper = interval
    changes = sorted({lower} | {int(r) + 1 for r in short.ravel() if lower <= r < upper})
    masks, starts = masks_for(tuple(map(tuple, short)), tuple(changes))
    packed = cube.reshape(len(cube), 20, 15, len(q))[:, 2:]
    coordinates = np.arange(15)[:, None] * NRESOURCE + q[None, :]
    contexts = {
        "short_tail": (short, masks, *[covariance(cube[g, -5:]) for g in frame_groups]),
        "extended_periodic": (
            coordinates, None,
            covariance(packed[:, train].mean(axis=1)),
            covariance(packed[:, reserved].mean(axis=1)),
        ),
    }
    return contexts, starts


def wrong_reference(cube, carriers, shift):
    slots = np.arange(2, 302)
    rotor = np.column_stack([pn_reference(int(k), slots + shift) for k in carriers])
    return np.asarray(cube, dtype=np.complex128) * rotor[None].conj()


def permute_lanes(template, rng):
    shuffled = template.copy()
    for lane in range(4):
        shuffled[lane::4] = rng.permutation(template[lane::4])
    return shuffled


def projectors_for_reference(ratio):
    """Remove constant and published empirical/analytic ratio directions."""
    projectors = []
    for column in ratio.T:
        basis = np.column_stack((np.ones(len(column)), column.real, column.imag))
        u, singular, _ = np.linalg.svd(basis, full_matrices=False)
        kept = u[:, singular > 1e-10]
        projectors.append(np.eye(len(column)) - kept @ kept.T)
    return np.stack(projectors)


def reference_adjusted_context(cube, ids, train, reserved, qin):
    slots = np.arange(2, 302)
    analytic = np.column_stack([pn_reference(int(k), slots) for k in ids])
    empirical = np.exp(.5j * np.pi * qin[ids[:, None], slots[None, :] - 1]).T
    cycle_ratio = (empirical * analytic.conj()).reshape(20, 15, len(ids))[2:]
    cycles = cube.reshape(len(cube), 20, 15, len(ids))[:, 2:, :, :]
    projectors, matrices, reference_example = [], [], []
    for indices in (train, reserved):
        ratio = cycle_ratio[indices].mean(axis=0)
        p = projectors_for_reference(ratio)
        y = cycles[:, indices].mean(axis=1)
        matrices.append(covariance(np.einsum("cst,ftc->fsc", p, y)))
        projectors.append(p)
        reference_example.append(covariance(ratio[None]))
    coordinates = np.arange(15)[:, None] * NRESOURCE + code_indices(ids)[None, :]
    return (coordinates, None, *matrices, projectors), (coordinates, None, *reference_example)


def test_windows(template, receiver, interval, observations):
    rng = np.random.default_rng(SEED + 1)
    shuffles = [permute_lanes(template, rng) for _ in range(CONTROLS)]
    train = receiver["calibration_cycles_after_first_two"]
    reserved = receiver["reserved_cycles_after_first_two"]
    qin = loadmat(QIN)["referenceTemplateRotations"]
    rows, control_scores = [], []
    with np.load(observations, allow_pickle=False) as archive:
        for window in receiver["receiver_windows"]:
            if window["duplicate_of"] is not None:
                continue
            name = window["dataset"].replace("-", "_") + "_" + window["array"]
            ids = archive[name + "_indices"]
            cube = archive[name + "_unknown"]
            if len(cube) != 14:
                raise ValueError("Unexpected receiver frame count")
            groups = np.arange(14).reshape(7, 2)
            groups = groups[np.random.default_rng(SEED).permutation(len(groups))]
            frame_groups = [groups[:3].ravel(), groups[3:].ravel()]
            aligned = wrong_reference(cube, ids, 0)
            shifted = wrong_reference(cube, ids, PN_WRONG_SHIFT)
            actual, starts = run_contexts(aligned, ids, frame_groups, train, reserved, interval)
            wrong, _ = run_contexts(shifted, ids, frame_groups, train, reserved, interval)
            projected, reference_only = reference_adjusted_context(aligned, ids, train, reserved, qin)
            wrong_projected, _ = reference_adjusted_context(shifted, ids, train, reserved, qin)
            actual["extended_reference_ratio_removed"] = projected
            wrong["extended_reference_ratio_removed"] = wrong_projected
            for mode in actual:
                if mode == "short_tail" and not starts:
                    rows.append({"window": name, "mode": mode, "status": "no_three_slot_carrier"})
                    continue
                result = template_test(template, *actual[mode])
                controls = [template_test(v, *actual[mode])["held_score"] for v in shuffles]
                rows.append({
                    "window": name, "unknown_carriers_unsigned": ids.tolist(), "mode": mode, "status": "evaluated",
                    "short_support_representatives": starts if mode == "short_tail" else None,
                    "wrong_pn_held_score": template_test(template, *wrong[mode])["held_score"],
                    "reference_only": template_test(template, *reference_only) if mode == "extended_periodic" else None,
                    **result,
                })
                control_scores.append((len(rows) - 1, mode, controls))
                print(name, mode, round(result["held_score"], 5), flush=True)
    for i, mode, values in control_scores:
        family = np.max([c for _, m, c in control_scores if m == mode], axis=0)
        v = rows[i]["held_score"]
        rows[i]["lane_preserving_controls"] = values
        rows[i]["family_rank_within_mode"] = float((1 + np.sum(family >= v - 1e-12)) / (1 + CONTROLS))
    return rows


def geometric_audit(template, starts, receiver):
    masks, geometry_metadata = band_masks()
    duplicates = {f"{row['dataset']}/{row['array']}" for row in receiver["receiver_windows"]
                  if row["duplicate_of"] is not None}
    result = []
    with np.load(WIDEBAND_IQ, allow_pickle=False) as evidence:
        for row in starts:
            start = row["template_refined_endpoint"]
            frame = row["candidate"]
            samples = evidence[f"frame_{frame}_symbols"]
            coarse = row["conditional_predicted_endpoint"] + row["modulation_endpoint_minus_prediction"]
            # The refined start differs from the raw constellation start in one frame.
            before = evidence[f"frame_{frame}_preceding_symbols"]
            full = np.r_[before, samples]
            origin = coarse - len(before)
            q = np.arange(start, NFRAME) % NRESOURCE
            choices = {}
            for name, carriers in masks.items():
                if name in duplicates:
                    continue
                coords = np.flatnonzero(np.isin(NONPILOT[q], carriers))
                decisions = full[start + coords - origin].real < 0
                choices[name] = {
                    **ambiguity(template, coords),
                    "projected_measured_sign_disagreements": int(np.sum(decisions != template[coords % PERIOD])),
                }
            result.append({"frame": frame, "start": start, "masks": choices})
    return result, geometry_metadata


def full_reference_audit(template, starts):
    slots = np.arange(2, 302)
    qin = loadmat(QIN)["referenceTemplateRotations"]
    indices = NONPILOT % 1024
    empirical = np.exp(.5j * np.pi * qin[indices[:, None], slots[None, :] - 1]).T
    analytic = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT])
    ratio = empirical * analytic.conj()
    order = np.arange(NFRAME).reshape(len(slots), NRESOURCE)
    signs = 1 - 2 * template.astype(int)
    errors = []
    for phase in range(PERIOD):
        expected = signs[(order + phase) % PERIOD]
        errors.append({
            "resource_phase": phase,
            "direct_disagreements": int(np.count_nonzero(abs(ratio - expected) > 1e-8)),
            "complement_disagreements": int(np.count_nonzero(abs(ratio + expected) > 1e-8)),
        })
    mismatches = np.count_nonzero(abs(ratio + signs[order % PERIOD]) > 1e-8, axis=1)
    last_bad = int(np.flatnonzero(mismatches)[-1]) if mismatches.any() else -1
    earliest = int(slots[last_bad + 1]) if last_bad + 1 < len(slots) else None
    return {
        "analytic_empirical_ratio_nonreal_resources": int(np.count_nonzero(abs(ratio.imag) > 1e-8)),
        "absolute_phase_and_polarity_candidates": errors,
        "absolute_phase0_complement_errors_per_slot": mismatches.tolist(),
        "first_exact_suffix_slot": earliest,
        "exact_suffix_resource_count": int(np.sum(slots >= earliest) * NRESOURCE) if earliest is not None else 0,
        "measured_wideband_tail_start_phases": [
            {"frame": row["candidate"], "absolute_resource_mod60": row["template_refined_endpoint"] % PERIOD,
             "sign_disagreements_if_absolute_phase": int(np.count_nonzero(
                 signs[np.arange(row["template_refined_endpoint"], NFRAME) % PERIOD]
                 != signs[np.arange(NFRAME - row["template_refined_endpoint"]) % PERIOD]
             ))}
            for row in starts
        ],
    }


def synthetic_control(template, interval):
    q = code_indices([525, 526, 527])
    r = short_slots(q)
    support = r >= interval[0]
    coordinates = np.arange(15)[:, None] * NRESOURCE + q[None, :]
    random = np.random.default_rng(SEED + 2)
    results = {}
    for mode, index, mask in (("short_tail", r, support[None]), ("extended_periodic", coordinates, None)):
        p = 17
        signal = 1 - 2 * template[(index + p) % PERIOD].astype(float)
        if mask is not None:
            signal *= mask[0]
        gain = np.exp(1j * random.uniform(-np.pi, np.pi, (14, 1, len(q))))
        noise = np.sqrt(.02) * (random.normal(size=(14, *index.shape)) + 1j * random.normal(size=(14, *index.shape)))
        y = gain * signal + (4 - 7j) + noise
        results[mode] = {
            "synthetic_phase": p,
            "with_template": template_test(template, index, mask, covariance(y[:6]), covariance(y[6:])),
            "noise_only": template_test(template, index, mask, covariance(noise[:6]), covariance(noise[6:])),
        }
    return results


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    require_run_context()
    wideband = read_json(WIDEBAND)
    folds = wideband["cycle_folds"]
    words = {fold["binary_template"] for fold in folds}
    if len(words) != 1 or any(fold["period"] != PERIOD for fold in folds):
        raise ValueError("Wideband template is not frozen")
    template = np.array([c == "1" for c in words.pop()])
    starts = wideband["conditional_length_audit"]["comparisons"]
    interval = tuple(int(op(row["template_refined_endpoint"] for row in starts)) for op in (min, max))
    receiver = read_json(RECEIVER)
    strong = read_json(GUARD_RECEIVER)
    geometry, geometry_metadata = geometric_audit(template, starts, receiver)
    band_inputs = [ROOT / TRACK, ROOT / WAV_MODEL]
    band_inputs.extend(SELECTION_ROOT / dataset / "selection.json" for dataset in ("DSX01-S05", "paired_RX"))
    results = {
        "schema": "starlink-narrowband-one-template-transfer/v1",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest(p)
                          for p in (NPZ, RECEIVER, GUARD_NPZ, GUARD_RECEIVER, WIDEBAND, WIDEBAND_IQ, QIN, *band_inputs)},
        "template": "".join(map(str, template.astype(int))),
        "seed": SEED, "control_seed": SEED + 1, "synthetic_seed": SEED + 2,
        "permutation_count": CONTROLS,
        "short_start_search_inclusive": interval,
        "short_frames_split": np.arange(14).reshape(7, 2)[np.random.default_rng(SEED).permutation(7)].tolist(),
        "extended_cycle_split_after_first_two": [receiver["calibration_cycles_after_first_two"], receiver["reserved_cycles_after_first_two"]],
        "wrong_reference_shift_slots": PN_WRONG_SHIFT,
        "conditional_geometry": geometry,
        "conditional_geometry_metadata": geometry_metadata,
        "front_end_guards_samples": [receiver["edge_guard_samples"], strong["edge_guard_samples"]],
        "full_reference_equivalence": full_reference_audit(template, starts),
        "measured_npz": test_windows(template, receiver, interval, NPZ),
        "measured_npz_larger_symbol_guard": test_windows(template, strong, interval, GUARD_NPZ),
        "synthetic_controls": synthetic_control(template, interval),
        "scope": "Finite shared-phase family per window, adjacent-pair or cycle splits. Exact known shape tested against shuffled shapes; no independent RF repetitions assumed. WAV geometry only.",
    }
    out = experiment_output("E026")
    out.mkdir(parents=True, exist_ok=True)
    (out / "narrowband_cycle_transfer.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
