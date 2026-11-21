"""Test the frozen 60-bit template on samples from the original WAV."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import RAW_WAV, experiment_output, require_run_context
from audit_h2_projection import TRACK, WAV_MODEL, band_masks
from audit_narrowband_cycle import (
    CONTROLS, NRESOURCE, PERIOD, WIDEBAND, covariance, masks_for, permute_lanes,
    projectors_for_reference, short_slots, template_test,
)
from layout_core import NONPILOT
from receiver_models import pn_reference

FOLD = ROOT / "experiments/E005/results/recovered_cyclic_waveform.json"
SEED = 2026112127
GUARDS = (.35, .75)
BLOCK_SECONDS = .2
FRAMES_PER_BLOCK = 16
SLOT_SECONDS = 4.4e-6
CP_SECONDS = 2 / 15 * 1e-6
GRID = 234375
SLOTS = np.arange(2, 302)
INJECTION_PHASE = 17


def select_frame_ids(block, first, last):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, block]))
    return np.sort(rng.choice(np.arange(first, last), FRAMES_PER_BLOCK, replace=False))


def fit_tones(samples, frequencies, fs):
    times = np.arange(samples.shape[-1]) / fs
    design = np.exp(2j * np.pi * times[:, None] * frequencies[None, :])
    if samples.shape[-1] < len(frequencies):
        raise ValueError("More unknown tones than raw samples in an OFDM slot")
    inverse = np.linalg.pinv(design, rcond=1e-9)
    return samples @ inverse.T, float(np.linalg.cond(design))


def raw_phase(coeff, time):
    a, b, c = coeff
    return 2 * np.pi * (a * time ** 3 / 3 + b * time ** 2 / 2 + c * time)


def extraction(raw, epochs, fs, clock, carriers, offset, coeff, guard, injection=None):
    """Fit native samples using an evenly spaced design and restore slot phase."""
    slots = epochs[:, None] + SLOTS[None, :] * SLOT_SECONDS * fs / clock
    first = np.ceil(slots + CP_SECONDS * fs / clock + guard).astype(np.int64)
    last = np.ceil(slots + SLOT_SECONDS * fs / clock - guard).astype(np.int64)
    counts = last - first
    if first.min() < 0 or last.max() > len(raw):
        raise IndexError("Incomplete WAV frame")
    tone_hz = carriers * GRID + offset
    phase_sync = (first - slots) * clock / fs
    slot_phase = (
        carriers[None, None, :] * GRID * (phase_sync[..., None] - CP_SECONDS)
        + offset * (phase_sync + SLOTS[None, :] * SLOT_SECONDS)[..., None]
    )
    phase_sync = np.exp(2j * np.pi * slot_phase)
    frequency_cw = np.polyval(coeff, np.median(epochs) / fs)
    tone_fit_hz = np.r_[tone_hz * clock, 0., -frequency_cw]
    out = np.empty((*first.shape, len(carriers)), dtype=np.complex128)
    conditions = []
    for count in np.unique(counts):
        mask = counts == count
        times = first[mask, None] + np.arange(count)
        data = raw[times].astype(float) / 32768
        data = (data[..., 0] + 1j * data[..., 1]) * np.exp(-1j * raw_phase(coeff, times / fs))
        if injection is not None:
            signal = injection[mask] * phase_sync[mask]
            increments = np.exp(2j * np.pi * np.arange(count)[:, None] * tone_hz[None] * clock / fs)
            data = data + signal @ increments.T
        fitted, condition = fit_tones(data, tone_fit_hz, fs)
        out[mask] = fitted[:, :len(carriers)] * phase_sync[mask].conj()
        conditions.append({"samples": int(count), "slots": int(mask.sum()), "condition": condition})
    return out, conditions


def context(cube, template, q):
    short = cube[:, -5:]
    periodic = cube.reshape(len(cube), 20, 15, len(q))[:, 2:].mean(axis=1)
    ordinal = np.arange(15)[:, None] * NRESOURCE + q[None, :]
    ratio = 1 - 2 * template[ordinal % PERIOD].astype(float)
    projectors = projectors_for_reference(ratio)
    stripped = np.einsum("cst,ftc->fsc", projectors, periodic)
    return {
        "short_tail": covariance(short),
        "extended_periodic": covariance(periodic),
        "extended_reference_ratio_removed": covariance(stripped),
    }


def branch_models(q, template, interval):
    short = short_slots(q)
    lower, upper = interval
    changes = sorted({lower} | {int(r) + 1 for r in short.ravel() if lower <= r < upper})
    masks, starts = masks_for(tuple(map(tuple, short)), tuple(changes))
    index = np.arange(15)[:, None] * NRESOURCE + q[None, :]
    ratio = 1 - 2 * template[index % PERIOD].astype(float)
    projector = projectors_for_reference(ratio)
    return {
        "short_tail": (short, masks, None),
        "extended_periodic": (index, None, None),
        "extended_reference_ratio_removed": (index, None, [projector, projector]),
    }, starts


def summarize_groups(template, models, block_tables, left_blocks, right_blocks, shuffles=()):
    results = {}
    for mode, (coordinates, masks, projector) in models.items():
        left, right = [
            np.stack([block_tables[b][mode] for b in group]).mean(axis=0)
            for group in (left_blocks, right_blocks)
        ]
        selection = template_test(template, coordinates, masks, left, right, projector)
        controls = [template_test(w, coordinates, masks, left, right, projector)["held_score"] for w in shuffles]
        results[mode] = {**selection, "controls": controls}
    return results


def short_selectivity(template, short_model, block_tables, wrong_tables, regions, shuffles):
    """Retain only the increment attributable to PN and event presence."""
    coordinates, masks, _ = short_model

    def contrasts(group, wanted):
        return np.stack([wanted[b]["short_tail"] for b in group]).mean(axis=0)

    on = []
    post = []
    for active, inactive in zip(regions["on"], regions["post"], strict=True):
        on.append(contrasts(active, block_tables) - contrasts(active, wrong_tables))
        post.append(contrasts(inactive, block_tables) - contrasts(inactive, wrong_tables))
    basis = [a - b for a, b in zip(on, post, strict=True)]
    data = template_test(template, coordinates, masks, *basis)
    null = [template_test(w, coordinates, masks, *basis)["held_score"] for w in shuffles]
    return {**data, "controls": null}


def inject_coefficients(template, q, rotor, count, gain_scale, short_start, seed):
    rng = np.random.default_rng(seed)
    ordinal = np.arange(len(SLOTS))[:, None] * NRESOURCE + q[None, :]
    signs = 1 - 2 * template[(ordinal + INJECTION_PHASE) % PERIOD].astype(float)
    if short_start is not None:
        signs *= ordinal >= short_start
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, (count, 1, len(q))))
    return gain_scale * signs[None, :, :] * rotor[None, :, :] * phase


def analyze_guards(raw, fs, clock, offset, model, coeff, template, q, carriers, interval, epoch_by_block):
    models, starts = branch_models(q, template, interval)
    rotor = np.column_stack([pn_reference(int(k), SLOTS) for k in carriers])
    wrong = np.column_stack([pn_reference(int(k), SLOTS + 1) for k in carriers])
    random = np.random.default_rng(SEED + 1)
    shuffles = [permute_lanes(template, random) for _ in range(CONTROLS)]
    regions = {
        "on": [model["calibration_blocks"], model["heldout_blocks"]],
        "post": [model["post_noise_calibration_blocks"], model["post_heldout_blocks"]],
    }
    post = sum(regions["post"], [])
    results, arrays = [], {}
    for guard in GUARDS:
        report = {"guard_samples": guard, "short_support_representatives": starts, "receiver_by_block": []}
        block_tables, wrong_tables, timing_tables = {}, {}, {}
        post_rms = []
        for b, epochs in epoch_by_block.items():
            y, diagnostics = extraction(raw, epochs, fs, clock, carriers, offset, coeff, guard)
            ypn = y * rotor[None].conj()
            block_tables[b] = context(ypn, template, q)
            wrong_tables[b] = context(y * wrong[None].conj(), template, q)
            shifted, _ = extraction(raw, epochs + SLOT_SECONDS / 4 * fs / clock,
                                    fs, clock, carriers, offset, coeff, guard)
            timing_tables[b] = context(shifted * rotor[None].conj(), template, q)
            rms = float(np.sqrt(np.mean(abs(y) ** 2)))
            if b in post:
                post_rms.append(rms)
            report["receiver_by_block"].append({"block": b, "coefficient_rms": rms, "slot_design": diagnostics})
        for region, groups in regions.items():
            report[region] = summarize_groups(template, models, block_tables, *groups, shuffles)
            report[region + "_wrong_pn"] = summarize_groups(template, models, wrong_tables, *groups)
            report[region + "_quarter_slot_epoch_delay"] = summarize_groups(template, models, timing_tables, *groups)
        report["pn_and_event_selective_short"] = short_selectivity(
            template, models["short_tail"], block_tables, wrong_tables, regions, shuffles
        )
        for mode in models:
            arrays[f"guard{int(guard * 100)}_{mode}"] = np.stack([r[mode] for r in block_tables.values()])
        for label, tables in (("wrong_pn", wrong_tables), ("epoch_delay", timing_tables)):
            key = f"guard{int(guard * 100)}_short_tail_{label}"
            arrays[key] = np.stack([r["short_tail"] for r in tables.values()])
        injections = []
        for type_, start in (("short_tail", interval[0]), ("extended_periodic", None)):
            for amplitude in (.25, .5, 1.):
                tables = {}
                scale = float(np.median(post_rms) * amplitude)
                for b in post:
                    epochs = epoch_by_block[b]
                    values = inject_coefficients(template, q, rotor, len(epochs), scale, start, SEED + b + 2)
                    recovered, _ = extraction(raw, epochs, fs, clock, carriers, offset, coeff, guard, values)
                    tables[b] = context(recovered * rotor[None].conj(), template, q)
                fitted = summarize_groups(template, {type_: models[type_]}, tables, *regions["post"])[type_]
                injections.append({"mode": type_, "amplitude_in_post_carrier_rms": amplitude,
                                   "coefficient_amplitude": scale, "phase": INJECTION_PHASE, **fitted})
        report["injections"] = injections
        results.append(report)
        print("guard", guard, {mode: report["on"][mode]["held_score"] for mode in models}, flush=True)
    for region in regions:
        for mode in models:
            maximum = np.max([r[region][mode]["controls"] for r in results], axis=0)
            for r in results:
                test = r[region][mode]
                test["per_mode_guard_adjusted_rank"] = float((1 + np.sum(maximum >= test["held_score"] - 1e-12)) / (1 + CONTROLS))
    mode = "pn_and_event_selective_short"
    maximum = np.max([r[mode]["controls"] for r in results], axis=0)
    for r in results:
        r[mode]["guard_adjusted_rank"] = float((1 + np.sum(maximum >= r[mode]["held_score"] - 1e-12)) / (1 + CONTROLS))
    return results, arrays


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    require_run_context()
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    if (fs, raw.shape[1], raw.dtype) != (2400000, 2, np.dtype("int16")):
        raise ValueError("Unexpected WAV format")
    model = read_json(ROOT / WAV_MODEL)
    fold = read_json(FOLD)
    clock = 1 - fold["inherited_phase_drift_samples_s"] / fs
    offset = model["exact"]["carrier_offset_from_moving_line_hz"]
    phase_samples = fs * (1e-6 * model["epoch_from_power_guard_phase_us"]
                          + model["exact"]["offset_from_power_guard_epoch_s"])
    template_record = read_json(WIDEBAND)
    template = np.array([b == "1" for b in template_record["cycle_folds"][0]["binary_template"]])
    ends = template_record["conditional_length_audit"]["comparisons"]
    interval = tuple(f(x["template_refined_endpoint"] for x in ends) for f in (min, max))
    masks, _ = band_masks()
    carriers = np.array(masks["wav"])
    q = np.flatnonzero(np.isin(NONPILOT, carriers))
    with np.load(ROOT / TRACK, allow_pickle=False) as track:
        coeff = track["quadratic_coefficients"]
    blocks = sorted(sum([model[k] for k in ("calibration_blocks", "heldout_blocks",
                                            "post_noise_calibration_blocks", "post_heldout_blocks")], []))
    epoch_by_block, frames_by_block = {}, {}
    for b in blocks:
        from_epoch = b * BLOCK_SECONDS * fs * clock
        to_epoch = (b + 1) * BLOCK_SECONDS * fs * clock
        first = max(0, int(np.ceil((from_epoch - phase_samples) / 3200)))
        last = int(np.floor((to_epoch - phase_samples) / 3200))
        indices = select_frame_ids(b, first, last)
        epoch_by_block[b] = (indices * 3200 + phase_samples) / clock
        frames_by_block[b] = indices.tolist()
    rows, arrays = analyze_guards(raw, fs, clock, offset, model, coeff, template, q, carriers, interval, epoch_by_block)
    result = {
        "schema": "starlink-wav-60-transfer/v1",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest(p)
                          for p in (RAW_WAV, ROOT / TRACK, ROOT / WAV_MODEL, FOLD, WIDEBAND)},
        "carriers_signed": carriers.tolist(), "nonpilot_ordinals": q.tolist(), "sample_rate_hz": fs,
        "carrier_offset_from_line_hz": offset, "frame_clock_ratio": clock, "epoch_phase_samples": phase_samples,
        "frame_seed": SEED, "permutation_seed": SEED + 1, "permutations": CONTROLS,
        "blocks": {k: model[k] for k in ("calibration_blocks", "heldout_blocks",
                                          "post_noise_calibration_blocks", "post_heldout_blocks")},
        "frames_by_block": frames_by_block, "short_search": interval,
        "fits": rows,
        "scope": "Direct complex raw samples, conditional known clock and carrier grid, per-carrier unknown gain. Injected controls are in-memory variations of post-event samples. No recovered message asserted.",
    }
    out = experiment_output("E027")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav_cycle_transfer.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "wav_block_correlations.npz", blocks=np.array(blocks), **arrays)


if __name__ == "__main__":
    main()
