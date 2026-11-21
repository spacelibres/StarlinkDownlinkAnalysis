"""Recover and verify a relative 60-symbol pattern in the user's original WAV."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from receiver_models import pn_reference


SEED = 2026112142
RATE = 2400000.
FRAME = 3200
SLOT = 4.4e-6
CP = 2e-6/15
GRID = 234375.
EPOCH = 1557.
COARSE_OFFSET = 117187.5
CARRIERS = np.array([-4, -3, 2, 3])
PERIOD = 15
HARMONIC_GROUP = 12
HARMONIC_RATE = RATE / FRAME
PN_NULLS = 399


def extract_local(mean, harmonics, offset):
    slots = np.arange(2, 302)
    n = FRAME * 32
    local_samples = 2.8 + np.linspace(0, 5.2, 8)
    indices = np.rint((EPOCH + RATE*SLOT*slots[:, None] + local_samples)*32).astype(int)
    absolute = (indices/32 - EPOCH)/RATE
    relative = absolute - SLOT*slots[:, None]
    out = []
    for i, carriers in enumerate(CARRIERS.reshape(2, -1)):
        f = np.zeros(n, dtype=complex)
        f[harmonics[i]] = mean[i]
        y = (fft.ifft(f) * n)[indices % n]
        design = np.exp(2j*np.pi*(relative[:, :, None]*GRID*carriers + offset*absolute[:, :, None]))
        out.append(np.einsum("sct,st->sc", np.linalg.pinv(design), y))
    z = np.column_stack(out)
    pn = np.column_stack([pn_reference(int(k), slots) for k in CARRIERS])
    return (z*pn.conj()).reshape(20, PERIOD, len(CARRIERS))


def local_advance(mean, harmonics):
    z = extract_local(mean, harmonics, COARSE_OFFSET)[2:]
    pairs = np.sum(z[1:] * z[:-1].conj(), axis=(0, 1))
    advance = np.angle(np.sum(pairs))
    side_angles = np.angle(pairs.reshape(2, -1).sum(axis=1))
    return float(advance), (side_angles/(2*np.pi*SLOT*PERIOD)).tolist()


def forward(harmonics, frequency_shift, *, period=PERIOD, pn_delta=0, first_slot=9):
    """Fourier series of rectangular CP-bearing candidate symbol intervals."""
    f = harmonics.ravel()*HARMONIC_RATE
    slots = np.arange(first_slot, 302)
    t = EPOCH/RATE + SLOT*slots
    e = np.exp(-2j*np.pi*f[:, None]*t)
    positions = (slots % period)[:, None] == np.arange(period)
    result = []
    for carrier in CARRIERS:
        pn = np.ones(len(slots)) if pn_delta is None else pn_reference(int(carrier), slots+pn_delta)
        ph = pn*np.exp(2j*np.pi*(frequency_shift*SLOT*slots - GRID*carrier*CP))
        q = (GRID*carrier + frequency_shift - f)*SLOT
        window = np.sinc(q)*np.exp(1j*np.pi*q)
        result.append((e*ph) @ positions * window[:, None])
    return np.hstack(result)


def harmonic_partition(harmonics, seed):
    rng = np.random.default_rng(seed)
    hold = []
    for row in harmonics:
        g = np.arange(len(row))//HARMONIC_GROUP
        group = rng.permutation(np.unique(g))
        hold.extend(np.isin(g, group[:len(group)//3]).tolist())
    return np.asarray(hold)


def coherence(reference, samples):
    norm = np.linalg.norm(reference)
    return abs(samples @ reference.conj())/np.maximum(np.linalg.norm(samples, axis=-1)*norm, 1e-30)


def measure(model, spectrum, fit, held_blocks, post_blocks, held_bins):
    selected = ~held_bins
    weights = np.linalg.lstsq(model[selected], fit[selected], rcond=1e-8)[0]
    predict = model[held_bins] @ weights
    y = spectrum[held_blocks][:, held_bins]
    background = spectrum[post_blocks][:, held_bins]
    return {"held_mean_coherence": float(coherence(predict, y.mean(axis=0))),
            "held_median_block_coherence": float(np.median(coherence(predict, y))),
            "post_mean_coherence": float(coherence(predict, background.mean(axis=0))),
            "parameters": int(model.shape[1]),
            "condition_number": float(np.linalg.cond(model[selected])),
            "held_coherence_by_block": coherence(predict, y).tolist()}, weights, predict


def axes_and_bits(training, reserved, held_mean):
    trains = training.reshape(len(CARRIERS), PERIOD)
    fitted = reserved.reshape(len(reserved), len(CARRIERS), PERIOD)
    withheld = held_mean.reshape(len(CARRIERS), PERIOD)
    axis = np.angle(np.sum(trains**2, axis=1))/2
    rot = np.exp(-1j*axis)
    rot *= np.where((trains[:, 0]*rot).real < 0, -1, 1)
    train_rot = trains * rot[:, None]
    thresholds = .4*np.median(abs(train_rot.real), axis=1)
    def labels(symbols):
        p = (symbols*rot[:, None]).real
        return np.where(abs(p) < thresholds[:, None], -1, p < 0).astype(int)
    cal, hold, all_blocks = (labels(v) for v in (trains, withheld, fitted))
    return {"calibration_axis_alignment": (abs(np.sum(trains**2, axis=1))/np.sum(abs(trains)**2, axis=1)).tolist(),
            "held_axis_alignment": (abs(np.sum(withheld**2, axis=1))/np.sum(abs(withheld)**2, axis=1)).tolist(),
            "projection_axes_rad": np.angle(rot.conj()).tolist(), "erase_thresholds": thresholds.tolist(),
            "calibration_bits_by_carrier_slot_class": cal.tolist(), "held_bits_by_carrier_slot_class": hold.tolist(),
            "held_block_erasures": (all_blocks < 0).sum(axis=(1, 2)).tolist(),
            "held_block_bit_disagreements": ((all_blocks >= 0) & (cal >= 0) & (all_blocks != cal)).sum(axis=(1, 2)).tolist()}


def phase_comparison(bits, held, reference, rng):
    ordinal = np.array([int(np.flatnonzero(NONPILOT == k)[0]) for k in CARRIERS])
    resource = (ordinal[:, None] + (np.arange(PERIOD)[None, :] - 2)*len(NONPILOT)) % 60
    if len(np.unique(resource)) != 60:
        raise ValueError("The four lanes no longer cover all relative classes")
    def compare(ref):
        alternatives = []
        for phase in range(60):
            expected = ref[(resource - phase) % 60]
            diff = (bits != expected) & (bits >= 0)
            same = (bits == expected) & (bits >= 0)
            parity = diff.sum(axis=1) > same.sum(axis=1)
            best = np.minimum(diff.sum(axis=1), same.sum(axis=1))
            adjusted = expected ^ parity[:, None]
            h = ((adjusted != held) & (held >= 0)).sum()
            alternatives.append({"phase_subtracted": phase, "calibration_differences": int(best.sum()),
                                 "held_differences": int(h), "carrier_complements": parity.tolist()})
        return sorted(alternatives, key=lambda t: (t["calibration_differences"], t["phase_subtracted"]))
    trials = []
    for _ in range(PN_NULLS):
        shuffled = reference.copy()
        for lane in range(4):
            shuffled[lane::4] = rng.permutation(shuffled[lane::4])
        trials.append(compare(shuffled)[0])
    return {"resource_index_by_lane_and_slot_class": resource.tolist(), "all_phases": compare(reference),
            "shuffled_best_fits": trials}


def slot_cycle(mean, harmonic_rows):
    x = np.zeros(FRAME, dtype=complex)
    x[harmonic_rows.ravel()] = mean
    wave = fft.ifft(x)
    power = abs(fft.fft(abs(wave)**2))/np.sum(abs(wave)**2)
    upper = int(380000/HARMONIC_RATE)
    lower = int(100000/HARMONIC_RATE)
    peak = lower+int(power[lower:upper+1].argmax())
    target = int(round(1/SLOT/HARMONIC_RATE))
    return {"strongest_hz_in_search": peak*HARMONIC_RATE, "target_bin_hz": target*HARMONIC_RATE,
            "normalized_peak": float(power[peak]), "target_normalized_coefficient": float(power[target])}


def main():
    require_run_context()
    inputs = {}
    previous = read(checked_artifact("v039-wav-joint-phase", "E041", "wav_joint_phase.json", inputs))
    waveform = checked_artifact("v039-wav-joint-phase", "E041", "wav_joint_phase.npz", inputs)
    train, held, post = (previous["groups"][part] for part in ("training", "held", "post"))
    with np.load(waveform, allow_pickle=False) as data:
        spectrum = data["mode_0_source_0_aligned_spectra"]
        harmonics = data["harmonics"]
    hold_bins = harmonic_partition(harmonics, SEED)
    calibration = spectrum[train].mean(axis=0)
    omega, separate = local_advance(calibration, harmonics)
    offset = COARSE_OFFSET + omega/(2*np.pi*SLOT*PERIOD)
    full = spectrum.reshape(len(spectrum), -1)
    variants = {"nominal": (offset, {}),
                "coarse_only": (COARSE_OFFSET, {}),
                "pn_shift_plus_one": (offset, {"pn_delta": 1}),
                "pn_shift_minus_one": (offset, {"pn_delta": -1}),
                "without_pn": (offset, {"pn_delta": None}),
                "period_14": (offset, {"period": 14}),
                "period_16": (offset, {"period": 16}),
                "alias_up": (offset+1/(SLOT*PERIOD), {}),
                "alias_down": (offset-1/(SLOT*PERIOD), {})}
    comparisons = {}
    for name, (trial_offset, settings) in variants.items():
        matrix = forward(harmonics, trial_offset, **settings)
        stats, fitted, pred = measure(matrix, full, calibration.ravel(), held, post, hold_bins)
        v = fitted.reshape(len(CARRIERS), -1)
        stats["axis_alignment"] = (abs(np.sum(v**2, axis=1))/np.sum(abs(v)**2, axis=1)).tolist()
        comparisons[name] = stats
        if name == "nominal":
            weights, forecast = fitted, pred
            inverse = np.linalg.pinv(matrix[hold_bins], rcond=1e-8)
            withheld = full[held][:, hold_bins] @ inverse.T
    bits = axes_and_bits(weights, withheld, withheld.mean(axis=0))
    # Use E025 only after all receive fits and relative decisions are frozen.
    external = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    sequence = np.asarray(external["cycle_folds"][0]["signed_template"]) < 0
    rng = np.random.default_rng(SEED+1)
    reference = phase_comparison(np.array(bits["calibration_bits_by_carrier_slot_class"]),
                                 np.array(bits["held_bits_by_carrier_slot_class"]), sequence, rng)
    synthetic = rng.choice((-1., 1.), size=60) * np.exp(1j*np.repeat(rng.uniform(-np.pi, np.pi, 4), 15))
    matrix = forward(harmonics, offset)
    fit = np.linalg.lstsq(matrix[~hold_bins], (matrix@synthetic)[~hold_bins], rcond=1e-8)[0]
    if not np.allclose(synthetic, fit, atol=1e-10):
        raise ValueError("Could not reconstruct the synthetic 60-value control")
    report = {"schema": "starlink-wav-tessellation/v1",
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
              "seed": SEED, "epoch_samples": EPOCH, "slot_seconds": SLOT, "carrier_indices": CARRIERS.tolist(),
              "coarse_offset_hz": COARSE_OFFSET, "calibration_phase_advance_rad": omega,
              "calibration_offset_by_side_hz": separate, "fitted_offset_hz": offset,
              "held_harmonics": int(hold_bins.sum()), "fit_harmonics": int((~hold_bins).sum()),
              "groups": previous["groups"], "models": comparisons, "relative_bits": bits,
              "reference_comparison": reference, "synthetic_max_coefficient_error": float(max(abs(fit-synthetic))),
              "slot_cycle": {key: slot_cycle(full[ids].mean(axis=0), harmonics)
                             for key, ids in (("calibration", train), ("held", held), ("post", post))},
              "scope": "Four receiver-relative polarities and cycle-frequency aliasing are retained. No new message integrity check."}
    output = experiment_output("E042")
    output.mkdir(parents=True, exist_ok=True)
    (output/"wav_tessellation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output/"wav_tessellation.npz", harmonic_rows=harmonics, held_harmonic_mask=hold_bins,
                        calibration_coefficients=weights, held_coefficients_by_block=withheld,
                        predicted_held_harmonics=forecast)
    print("Held spectrum coherence", comparisons["nominal"]["held_mean_coherence"], flush=True)


if __name__ == "__main__":
    main()
