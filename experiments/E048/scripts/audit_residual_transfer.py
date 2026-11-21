"""Cross-validate the parts that vary after removal of repeated low-entropy shapes."""

from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_narrowband_cycle import QIN, code_indices, permute_lanes, wrong_reference
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_tessellation import forward
from receiver_models import pn_reference


SEED = 2026112148


def coherence(a, b):
    return float(np.vdot(a, b).real/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))


def remove_subspace(data, columns):
    u, singular, _ = np.linalg.svd(columns, full_matrices=False)
    basis = u[:, singular > singular[0]*1e-9]
    return data-(data @ basis.conj()) @ basis.T


def wav_modes(reference, harmonics, hold_bins, c, offset, spectra, groups):
    train, held, post = [np.sort(groups[p]) for p in ("training", "held", "post")]
    design = forward(harmonics, offset)
    common = (design @ c).reshape(harmonics.shape)
    choices, arrays = {}, {}
    if not np.array_equal(reference["groups"]["held"], groups["held"]):
        raise ValueError("E042 holdouts disagree")
    for key, data in spectra.items():
        results = {}
        for degree in (0, 1, 2):
            modes, coordinates, bands = [], [], []
            for side, mean in enumerate(data[train].mean(axis=0)):
                k = harmonics[side]/1000
                nuisance = np.column_stack([common[side], *[k**i*mean for i in range(degree+1)]])
                residual = remove_subspace(data[:, side], nuisance)
                _, sv, shape = np.linalg.svd(residual[train], full_matrices=False)
                mode = shape[0]
                coeff = residual @ mode.conj()
                mask = hold_bins[side]
                shape_rows = design.reshape(*harmonics.shape, -1)[side]
                w = np.linalg.lstsq(shape_rows[~mask], mode[~mask], rcond=1e-8)[0]
                predicted = shape_rows[mask] @ w
                bands.append({"held_residual_energy_fraction_median": float(
                                  np.median(np.sum(abs(residual[held])**2, axis=1)/
                                            np.sum(abs(data[held, side])**2, axis=1))),
                              "leading_training_residual_fraction": float(sv[0]**2/np.sum(sv**2)),
                              "period15_mode_on_held_harmonics": float(
                                  abs(np.vdot(predicted, mode[mask]))/
                                  max(np.linalg.norm(predicted)*np.linalg.norm(mode[mask]), 1e-30)),
                              "held_leading_coefficient_median": float(np.median(abs(coeff[held]))),
                              "post_leading_coefficient_median": float(np.median(abs(coeff[post])))})
                modes.append(mode)
                coordinates.append(coeff)
            left, right = coordinates
            fit = np.vdot(left[train], right[train])/np.vdot(left[train], left[train])
            estimate = fit*left[held]
            target = right[held]
            control = [coherence(estimate, np.roll(target, i)) for i in range(1, len(held))]
            error = np.linalg.norm(target-estimate)**2 / np.linalg.norm(target)**2
            results[str(degree)] = {"held_score": coherence(estimate, target), "time_rotation_controls": control,
                                    "held_nmse_relative_to_zero": float(error),
                                    "post_score": coherence(fit*left[post], right[post]),
                                    "training_gain": [float(fit.real), float(fit.imag)], "bands": bands}
            arrays[f"{key}_degree{degree}_modes"] = np.stack(modes)
            arrays[f"{key}_degree{degree}_coefficients"] = np.stack(coordinates)
        choices[key] = results
    return {"groups": groups, "frontends": choices}, arrays


def paired_score(residual_a, residual_b):
    possible = list(permutations(range(len(residual_b)//2)))
    block = residual_b.reshape(len(possible[0]), 2, *residual_b.shape[1:])
    return [coherence(residual_a, block[list(order)].reshape(residual_b.shape)) for order in possible]


def npz_window(cube, carriers, rotations, groups, cycles):
    slots = np.arange(2, 302)
    analytic = np.column_stack([pn_reference(int(k), slots) for k in carriers])
    empirical = np.exp(.5j*np.pi*rotations[carriers[:, None], slots[None, :]-1]).T
    ratio = (empirical*analytic.conj()).reshape(20, 15, len(carriers))[2:]
    observed = wrong_reference(cube, carriers, 0).reshape(14, 20, 15, len(carriers))[:, 2:]
    halves = [(observed[:, ids].mean(axis=1), ratio[ids].mean(axis=0)) for ids in cycles]
    scores, keep = [], []
    for fit, test in (groups, groups[::-1]):
        side = []
        for values, known in halves:
            mean = values[fit].mean(axis=0)
            picks = []
            for k in range(len(carriers)):
                reference = known[:, k]
                controls = np.column_stack([np.ones(15), np.arange(15)-7, mean[:, k],
                                            reference, reference.conj()])
                picks.append(remove_subspace(values[test, :, k], controls))
            side.append(np.stack(picks, axis=-1))
        scores.append(paired_score(*side))
        keep.append(np.stack(side))
    null = (np.array(scores[0])[:, None]+np.array(scores[1])[None, :]).ravel()[1:]/2
    return {"score_by_direction": [x[0] for x in scores],
            "pair_permutations_by_direction": scores,
            "overall_score": float(np.mean([x[0] for x in scores])),
            "joint_nonidentity_controls": null.tolist()}, keep


def npz_modes(frontends, rotations, rng):
    random_order = rng.permutation(18)
    chronological = np.arange(18)
    order = rng.permutation(7)
    frame_groups = np.arange(14).reshape(7, 2)[order]
    groups = (frame_groups[:3].ravel(), frame_groups[3:].ravel())
    methods = {tag: [x[:9], x[9:]] for tag, x in (("random_cycles", random_order), ("early_late", chronological))}
    all_results, arrays = {}, {}
    for name, path, report in frontends:
        windows = {tag: [] for tag in methods}
        with np.load(path, allow_pickle=False) as archive:
            for row in report["receiver_windows"]:
                if row["duplicate_of"] is not None:
                    continue
                window = f'{row["dataset"]}_{row["array"]}'.replace("-", "_")
                ids = archive[f"{window}_indices"]
                cube = archive[f"{window}_unknown"]
                for tag, cycles in methods.items():
                    out, data = npz_window(cube, ids, rotations, groups, cycles)
                    windows[tag].append({"window": window, "carriers": ids.tolist(), **out})
                    for i, pair in enumerate(data):
                        arrays[f"{name}_{tag}_{window}_fold{i}"] = pair
        for collection in windows.values():
            collective = np.max([e["joint_nonidentity_controls"] for e in collection], axis=0)
            for w in collection:
                w["fraction_family_controls_ge_observed"] = float(np.mean(collective >= w["overall_score"]))
                w["max_family_control"] = float(max(collective))
        all_results[name] = windows
    return {"frame_groups": [g.tolist() for g in groups], "cycle_groups": {
        key: [v.tolist() for v in method] for key, method in methods.items()}, "frontends": all_results}, arrays


def wide_tail(report, archive, rng):
    template = np.asarray(report["cycle_folds"][0]["signed_template"])
    bank = np.stack([np.roll(template, shift) for shift in range(60)])
    rows, absolute, corrected = [], [], []
    for frame in report["signal_results"]:
        candidate = frame["candidate"]
        first = frame["scales"]["0.06"]["first_binary_resource"]
        values = archive[f"frame_{candidate}_symbols"].real
        chunks = np.arange(len(values)//120)
        chunks = rng.permutation(chunks)
        train, held = np.array_split(chunks, 2)
        means = []
        for g in (train, held):
            positions = (g[:, None]*120+np.arange(120)).ravel()
            phase = (first+positions) % 60
            means.append(np.bincount(phase, weights=values[positions], minlength=60)/(2*len(g)))
        match = means[0] @ bank.T
        winner = int(np.argmax(match))
        rows.append({"candidate": candidate, "groups_120_symbols": [train.tolist(), held.tolist()],
                     "calibrated_phase": winner, "held_score": coherence(bank[winner], means[1]),
                     "held_decision_differences": int(np.count_nonzero((bank[winner]<0) != (means[1]<0)))})
        absolute.append(means[1])
        corrected.append(np.roll(means[1], -winner))
    def metric(x):
        return float(np.linalg.norm(np.mean(x, axis=0))/np.mean(np.linalg.norm(x, axis=1)))
    return {"rows": rows, "fixed_index_average_ratio": metric(absolute),
            "after_per_frame_phase_ratio": metric(corrected)}


def cycle_phase_test(pipes, arrays, template, rng):
    """Choose a reference phase on one cycle half and predict the other half."""
    words = np.stack([template, *[permute_lanes(template, rng) for _ in range(399)]])
    phase = np.arange(60)[:, None, None]
    per_window, stability = {}, {}
    for frontend, schemes in pipes["frontends"].items():
        per_window[frontend] = {}
        for method, windows in schemes.items():
            results = {}
            for window in windows:
                name = window["window"]
                ordinals = code_indices(window["carriers"])
                coordinates = (np.arange(15)[:, None]*1004+ordinals[None, :]+phase) % 60
                raw = words[:, coordinates].astype(float)
                raw -= raw.mean(axis=2, keepdims=True)
                bank = raw/np.maximum(np.linalg.norm(raw, axis=2, keepdims=True), 1e-30)
                trials, selections = [], []
                for fold in range(2):
                    x, y = arrays[f"{frontend}_{method}_{name}_fold{fold}"]
                    fit = np.einsum("tjsc,fsc->tfjc", bank, x)
                    energy = (abs(fit)**2).sum(axis=-1)
                    chosen = energy.argmax(axis=-1)
                    index = (np.arange(len(words))[:, None], np.arange(len(x))[None, :], chosen)
                    prediction = fit[index][:, :, None, :]*bank[np.arange(len(words))[:, None], chosen]
                    numerator = np.einsum("tfsc,fsc->t", prediction.conj(), y).real
                    denominator = np.sqrt(np.sum(abs(prediction)**2, axis=(1, 2, 3)))*np.linalg.norm(y)
                    trials.append(numerator/np.maximum(denominator, 1e-30))
                    fixed = int(energy[0].sum(axis=0).argmax())
                    held_energy = (abs(np.einsum("jsc,fsc->fjc", bank[0], y))**2).sum(axis=-1)
                    a, b = chosen[0], held_energy.argmax(axis=-1)
                    def ties(values):
                        return np.isclose(values, values.max(axis=1, keepdims=True), rtol=1e-10, atol=0).sum(axis=1)
                    at = ties(energy[0])
                    bt = ties(held_energy)
                    phase_map = stability.setdefault(name, {})
                    for i, frame in enumerate(pipes["frame_groups"][1-fold]):
                        frame_votes = phase_map.setdefault(frame, [])
                        frame_votes.extend([int(a[i])] if at[i] == 1 else [None])
                        frame_votes.extend([int(b[i])] if bt[i] == 1 else [None])
                    selections.append({"source_phases_added": a.tolist(), "target_independent_phases_added": b.tolist(),
                                       "source_ties": at.tolist(), "target_ties": bt.tolist(),
                                       "matched_unique_phases": int(np.count_nonzero((a == b) & (at == 1) & (bt == 1))),
                                       "fixed_phase_selected": fixed,
                                       "fixed_phase_held_score": coherence(fit[0, :, fixed, None, :]*bank[0, fixed], y),
                                       "per_frame_held_score": float(trials[-1][0])})
                combined = np.mean(trials, axis=0)
                results[name] = {"folds": selections, "observed_score": float(combined[0]),
                                 "randomized_words": combined[1:].tolist()}
            family = np.max([r["randomized_words"] for r in results.values()], axis=0)
            for r in results.values():
                r["family_control_maximum"] = float(max(family))
                r["family_controls_ge_observed"] = int(np.count_nonzero(family >= r["observed_score"]))
            per_window[frontend][method] = results
    agreed = {}
    for name, cases in stability.items():
        agreed[name] = {str(i): values[0] if None not in values and len(set(values)) == 1 else None
                        for i, values in sorted(cases.items())}
    return {"windows": per_window, "phase_consensus_across_all_paths": agreed}


def main():
    require_run_context()
    inputs = {}
    def artifact(run, exp, file):
        return checked_artifact(run, exp, file, inputs)
    r41 = "v039-wav-joint-phase"
    r42 = "v040-wav-tessellation"
    mode_keys = [f"mode_{i}_source_0_aligned_spectra" for i in (0, 1)]
    w = read(artifact(r41, "E041", "wav_joint_phase.json"))
    t = read(artifact(r42, "E042", "wav_tessellation.json"))
    with np.load(artifact(r41, "E041", "wav_joint_phase.npz"), allow_pickle=False) as s, np.load(
            artifact(r42, "E042", "wav_tessellation.npz"), allow_pickle=False) as p:
        wav, wav_arrays = wav_modes(t, s["harmonics"], p["held_harmonic_mask"].reshape(s["harmonics"].shape),
                                    p["calibration_coefficients"], t["fitted_offset_hz"],
                                    {key: s[key] for key in mode_keys}, w["groups"])
    frontends = []
    for tag, run in (("guard06", "v006-guard06-variance"), ("guard15", "v006-guard15-variance")):
        f = artifact(run, "E008", "npz_crossvalidated_observations.npz")
        frontends.append((tag, f, read(artifact(run, "E008", "npz_information_audit.json"))))
    inputs[QIN] = sha256(QIN)
    rotations = loadmat(QIN)["referenceTemplateRotations"]
    rng = np.random.default_rng(SEED)
    npz, npz_arrays = npz_modes(frontends, rotations, rng)
    wide_run = "v023-binary-tail-final"
    meta = read(artifact(wide_run, "E025", "ut_binary_tail.json"))
    with np.load(artifact(wide_run, "E025", "ut_binary_tail.npz"), allow_pickle=False) as raw:
        wide = wide_tail(meta, raw, rng)
    npz["per_frame_reference_phase"] = cycle_phase_test(
        npz, npz_arrays, np.asarray(meta["cycle_folds"][0]["signed_template"]), rng)
    output = experiment_output("E048")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "starlink-residual-transfer/v1", "seed": SEED,
               "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
               "wav": wav, "npz": npz, "wide_tail_positive_control": wide,
               "scope": "Exploratory transfer evidence in three separate observables. Frame, cycle and block counts are not independent captures."}
    (output/"residual_transfer.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output/"residual_transfer.npz", **wav_arrays, **npz_arrays)
    print("Residual transfer computed on WAV, deduplicated NPZ and measured wideband tail", flush=True)


if __name__ == "__main__":
    main()
