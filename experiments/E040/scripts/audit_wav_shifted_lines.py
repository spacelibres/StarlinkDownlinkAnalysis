"""Separate an offset spectral grid from the CW in the original WAV."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import wavfile
from scipy.signal.windows import hann

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import RAW_WAV, experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_transfer import raw_phase


SEED = 2026112140
PERIOD = 3200
DELTAS = np.arange(-370., 375., 5.)
SPACINGS = np.arange(43750., 44251., 10.)
HARMONICS = np.array([-np.arange(427, 1400), np.arange(427, 1400)])
FAMILIES = ("frequency_corrected", "cw_phase_corrected", "receiver", "wrong_period_low", "wrong_period_high")
CONTROL_COUNT = 399


def comb_profile(power, frequencies, spacing):
    centers = spacing * HARMONICS
    local = np.interp(centers[:, :, None] + DELTAS, frequencies, power)
    return (local / np.maximum(local.mean(axis=2, keepdims=True), 1e-30)).mean(axis=1)


def nearest_tone_profile(power, frequencies):
    harmonics = np.r_[np.arange(-8, 0), np.arange(1, 9)]
    centers = harmonics[:, None] * 44000.0
    backgrounds = np.interp(centers + np.arange(-2300, 2301, 25), frequencies, power)
    ratio = np.interp(harmonics[:, None] * SPACINGS, frequencies, power)
    return (ratio / np.maximum(backgrounds.mean(axis=1, keepdims=True), 1e-30)).mean(axis=0)


def local_peak(profile):
    i = int(np.argmax(profile))
    center = DELTAS[i]
    if 0 < i < len(profile)-1:
        left, middle, right = profile[i-1:i+2]
        curvature = left + right - 2 * middle
        if curvature < 0:
            center += 2.5 * (left-right) / curvature
    return float(center)


def fold(residual, center_time, coordinate, delta):
    y = residual * np.exp(-2j * np.pi * delta * center_time)
    base = np.floor(coordinate).astype(int)
    fraction = coordinate - base
    neighbor = (base+1) % PERIOD
    def bin_sum(v):
        return (np.bincount(base, weights=v*(1-fraction), minlength=PERIOD) +
                np.bincount(neighbor, weights=v*fraction, minlength=PERIOD))
    weights = bin_sum(np.ones(y.shape))
    return (bin_sum(y.real) + 1j*bin_sum(y.imag)) / weights


def references(raw, fs, coefficients, phase_times, phase, spacing):
    n = fs//5
    j = np.arange(n)
    f = fft.fftshift(fft.fftfreq(n, 1/fs))
    win = hann(n, sym=False)
    banks, all_folds, fine_offsets, coarse_tracks = [], [], [], []
    for first in range(0, len(raw)-n+1, n):
        sample = raw[first:first+n].astype(np.float64) / 32768
        t = (first+j) / fs
        z = sample[:, 0] + 1j*sample[:, 1]
        oscillator = np.exp(-1j*raw_phase(coefficients, t))
        detrended = z*oscillator
        tracked = detrended*np.exp(-1j*np.interp(t, phase_times, phase))
        fft_power = [abs(fft.fftshift(fft.fft(data*win)))**2 for data in (detrended, tracked, z)]
        profiles = [comb_profile(values, f, spacing) for values in fft_power]
        profiles.extend(comb_profile(fft_power[0], f, spacing+offset) for offset in (-3, 3))
        banks.append(profiles)
        coarse_tracks.append(nearest_tone_profile(fft_power[0], f))
        deltas = [[local_peak(p) for p in profiles[k]] for k in (0, 1)]
        fine_offsets.append(deltas)
        frame = ((first+j)*(spacing/750)) % PERIOD
        dt = t - t.mean()
        all_folds.append([[fold(data, dt, frame, d) for d in (ds[1], ds[0], 0.0, -ds[1], -ds[0])]
                          for data, ds in zip((detrended, tracked), deltas, strict=True)])
    return np.asarray(banks), np.asarray(fine_offsets), np.asarray(all_folds), np.asarray(coarse_tracks)


def normalized_coefficients(blocks, band):
    y = fft.fft(blocks, axis=-1)[:, band]
    return y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-30)


def phase_shape(wave, training, held, post, band, shifts):
    vector = normalized_coefficients(wave, band)
    _, singular, rows = np.linalg.svd(vector[training], full_matrices=False)
    template = rows[0]
    projections = abs(vector @ template.conj())
    spectrum = np.zeros((len(held), PERIOD), dtype=complex)
    spectrum[:, band] = vector[held] * template.conj()
    shift_corr = abs(fft.ifft(spectrum, axis=1))*PERIOD
    shifted = shift_corr[np.arange(len(held)), shifts]
    null = np.median(shifted, axis=1)
    measured = float(np.median(projections[held]))
    return {"training_median": float(np.median(projections[training])),
            "held_median": measured, "post_median": float(np.median(projections[post])),
            "largest_training_singular": float(singular[0]), "shift_null_95pct": float(np.quantile(null, .95)),
            "shift_null_ge_held": int(np.count_nonzero(null >= measured))}, projections


def evaluate(profiles, fine_offsets, wave, comb44, train, held, post):
    rng = np.random.default_rng(SEED)
    swaps = np.array([rng.permutation(held) for _ in range(CONTROL_COUNT)])
    shifts = rng.integers(0, PERIOD, (CONTROL_COUNT, len(held)))
    summary = {}
    projections = np.empty((2, 2, 3, len(profiles)))
    for branch, key in enumerate(FAMILIES):
        x = profiles[:, branch]
        fixed = [local_peak(s) for s in x[train].mean(axis=0)]
        branch_out = {"trained_global_offsets_hz": fixed,
                      "held_global_offset_score": [float(np.interp(d, DELTAS, p))
                                                   for d, p in zip(fixed, x[held].mean(axis=0), strict=True)],
                      "post_global_offset_score": [float(np.interp(d, DELTAS, p))
                                                   for d, p in zip(fixed, x[post].mean(axis=0), strict=True)]}
        summary[key] = branch_out
        if branch >= 2:
            continue
        estimate = fine_offsets[:, branch]
        branch_out["held_offset_rms_difference_hz"] = float(np.sqrt(np.mean(np.diff(estimate[held], axis=1)**2)))
        branch_out["held_offset_correlation"] = float(np.corrcoef(estimate[held].T)[0, 1])
        branch_out["band_transfer"] = []
        for band_id, (band, fold_id) in enumerate(zip(HARMONICS, (0, 1), strict=True)):
            other = 1-band_id
            scores = lambda ids, offsets: np.array([np.interp(delta, DELTAS, x[i, band_id])
                                                    for i, delta in zip(ids, offsets, strict=True)])
            paired = scores(held, estimate[held, other])
            permutation = np.array([np.median(scores(held, estimate[ids, other])) for ids in swaps])
            entries = {"median_held_from_other_band": float(np.median(paired)),
                       "median_held_at_zero": float(np.median(scores(held, np.zeros(len(held))))),
                       "median_held_at_opposite_offset": float(np.median(scores(held, -estimate[held, other]))),
                       "median_post_from_other_band": float(np.median(scores(post, estimate[post, other]))),
                       "offset_permutation_ge_observed": int(np.count_nonzero(permutation >= np.median(paired))),
                       "offset_permutation_95pct": float(np.quantile(permutation, .95))}
            modes = (fold_id, 2, 3+fold_id)
            entries["coherent_fold"] = {}
            for i, (name, mode) in enumerate(zip(("transferred_offset", "no_offset", "opposite_offset"), modes, strict=True)):
                metrics, curve = phase_shape(wave[:, branch, mode], train, held, post, band, shifts)
                entries["coherent_fold"][name] = metrics
                projections[branch, band_id, i] = curve
            branch_out["band_transfer"].append(entries)
    best_44 = int(np.argmax(comb44[train].mean(axis=0)))
    summary["44khz"] = {"trained_spacing_hz": float(SPACINGS[best_44]),
                       "training_mean": float(comb44[train, best_44].mean()),
                       "held_mean": float(comb44[held, best_44].mean()),
                       "post_mean": float(comb44[post, best_44].mean())}
    return summary, projections, swaps, shifts


def synthetic_control(fs, spacing):
    rng = np.random.default_rng(SEED+1)
    n = fs//5
    t = np.arange(n)/fs
    frequencies = fft.fftshift(fft.fftfreq(n, 1/fs))
    noise = rng.normal(size=n) + 1j*rng.normal(size=n)
    wave = noise.copy()
    for tone in np.r_[spacing * HARMONICS[:, ::40].ravel()-273.2, 44000*np.r_[np.arange(-8, 0), np.arange(1, 9)]]:
        wave += .25*np.exp(1j*(2*np.pi*tone*t + rng.uniform(-np.pi, np.pi)))
    power = abs(fft.fftshift(fft.fft(wave*hann(n, sym=False))))**2
    peaks = [local_peak(p) for p in comb_profile(power, frequencies, spacing)]
    fitted_44 = SPACINGS[nearest_tone_profile(power, frequencies).argmax()]
    if max(abs(np.array(peaks)+273.2)) > 5 or abs(fitted_44-44000) > 10:
        raise ValueError("Synthetic grid estimates did not find the injected parameters")
    return {"expected_offset_hz": -273.2, "measured_offsets_hz": peaks,
            "expected_second_spacing_hz": 44000, "measured_second_spacing_hz": float(fitted_44)}


def main():
    require_run_context()
    track = ROOT / "experiments/E001/results/refined_track.npz"
    clock = ROOT / "experiments/E005/results/recovered_cyclic_waveform.json"
    phase = ROOT / "experiments/E006/results/wav_coherent_frame_probe.npz"
    inputs = {}
    wide = read(checked_artifact("v022-ut-replay", "E014", "ut_transfer.json", inputs))
    for path in (RAW_WAV, track, clock, phase):
        with path.open("rb") as stream:
            inputs[path] = hashlib.file_digest(stream, "sha256").hexdigest()
    timing = read(clock)
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    if fs != 2400000 or raw.shape[1] != 2 or raw.dtype != np.int16:
        raise ValueError("Expected original 2.4 MS/s int16 IQ")
    spacing = (fs - timing["inherited_phase_drift_samples_s"]) / PERIOD
    with np.load(track, allow_pickle=False) as t, np.load(phase, allow_pickle=False) as p:
        array = references(raw, fs, t["quadratic_coefficients"], p["carrier_time_s"], p["carrier_phase"], spacing)
    profiles, offsets, wave, comb44 = array
    train = np.asarray(timing["calibration_blocks"])
    held = np.asarray(timing["heldout_blocks"])
    post = np.asarray(timing["post_blocks"])
    models, projections, swaps, shifts = evaluate(profiles, offsets, wave, comb44, train, held, post)
    epochs = np.array([r["pss"]["epoch"]/480e6 for r in wide["candidates"] if r["complete_frame"]])
    count = np.rint((epochs-epochs[0])*750)
    wide_period_s, t0 = np.polyfit(count, epochs, 1)
    result = {"schema": "starlink-wav-shifted-lines/v1",
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): d for path, d in inputs.items()},
              "frequency_grid_hz": spacing, "frame_samples_at_nominal_rate": PERIOD,
              "block_seconds": .2, "bands_hz": [(int(x.min()*spacing), int(x.max()*spacing)) for x in HARMONICS],
              "calibration_blocks": train.tolist(), "heldout_blocks": held.tolist(), "post_blocks": post.tolist(),
              "seed": SEED, "controls": CONTROL_COUNT, "models": models,
              "wideband_comparator": {"fitted_frame_rate_hz": float(1/wide_period_s),
                                     "max_epoch_residual_ns": float(max(abs(epochs-t0-count*wide_period_s))*1e9),
                                     "complete_frames": len(epochs)},
              "synthetic": synthetic_control(fs, spacing)}
    out = experiment_output("E040")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav_shifted_lines.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "wav_shifted_lines.npz", deltas=DELTAS, spacings=SPACINGS, profiles=profiles,
                        offsets=offsets, folded=wave.astype(np.complex64), projections=projections,
                        comb44=comb44, swaps=swaps, shifts=shifts, centers_s=.2*np.arange(len(wave))+.1)
    print("Evaluated", len(wave), "WAV blocks and", len(epochs), "complete wideband timing references", flush=True)


if __name__ == "__main__":
    main()
