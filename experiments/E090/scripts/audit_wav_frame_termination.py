"""Track the terminal edge of frozen WAV templates without using band power alone."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import RAW_WAV, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_transfer import raw_phase
from audit_wav_shifted_lines import PERIOD
from audit_wav_tessellation import forward


SEED = 2026112191
ANCHOR = 13.635
WAV_INTERVAL = (12.5, 14.1)
FIT_INTERVAL = (13.35, 13.95)
EDGE_INTERVAL = (13.55, 13.72)
LEVEL_WINDOWS = ((13.45, 13.58), (13.73, 13.90))
BLOCKS = (8, 16, 32)
BOOTSTRAP = 499
SHIFT_CONTROLS = 39
OFFSETS = (0, 800, 1600, 2400)
PRIMARY = ("negative_matched", "positive_matched", "moving_cw")


def unit_rows(values):
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def in_interval(time, bounds):
    return (time >= bounds[0]) & (time < bounds[1])


def step_bank(sample_centers, edge_times, candidates):
    trend = np.column_stack((np.ones(len(sample_centers)), sample_centers - ANCHOR))
    regressor = (sample_centers[None, :] < edge_times[candidates, None]).astype(float)
    residual = regressor - (regressor @ np.linalg.pinv(trend).T) @ trend.T
    return regressor, trend, unit_rows(residual)


def step_fit(values, step, trend, contrasts):
    improvement = np.maximum(contrasts @ values, 0.)**2
    idx = improvement.argmax(axis=0)
    fitted = []
    for j, i in enumerate(idx):
        design = np.column_stack((trend, step[i]))
        beta = np.linalg.lstsq(design, values[:, j], rcond=None)[0]
        fitted.append(design @ beta)
    return idx, np.column_stack(fitted)


def inspect_profiles(raw, fs, starts, grid, delay, cw, offsets, harmonics, templates, phase_controls):
    primary, shape, null = [], [], []
    receiver = []
    raw_f = fft.fftfreq(PERIOD, 1 / fs)
    sample_index = np.arange(PERIOD)
    fixed_13 = in_interval(raw_f, (12000, 14001))
    fixed_center = in_interval(raw_f, (-100000, 100000)) & ~fixed_13
    lobe_bands = [in_interval(raw_f, band) for band in ((-950000, -650000), (500000, 750000))]
    for first in range(0, len(starts), 64):
        initial = starts[first:first + 64]
        ix = initial[:, None] + sample_index
        times = ix / fs
        z = raw[ix].astype(float) / 32768
        x = z[..., 0] + 1j * z[..., 1]
        base = x * np.exp(-1j * raw_phase(cw, times))
        frame_phase = initial * PERIOD * grid / fs - np.polyval(delay, (initial + PERIOD / 2) / fs)
        side_scores = []
        for side in range(2):
            corrected = base * np.exp(-2j * np.pi * offsets[side] * sample_index / fs)
            values = fft.fft(corrected, axis=1)[:, harmonics[side]] / PERIOD
            values *= np.exp(-2j * np.pi * (frame_phase % PERIOD)[:, None] * harmonics[side] / PERIOD)
            side_scores.append(abs(values @ templates[side].conj().T)**2)
        scores = np.stack(side_scores, axis=1)
        primary.append(np.column_stack((scores[:, :, 0], abs(base.mean(axis=1))**2)))
        shape.append(scores[:, :, 1])
        null.append(scores[:, :, 2:])
        raw_power = abs(fft.fft(x, axis=1) / PERIOD)**2
        demod_power = abs(fft.fft(base, axis=1) / PERIOD)**2
        receiver.append(np.column_stack((raw_power[:, fixed_center].sum(axis=1), raw_power[:, fixed_13].sum(axis=1),
                                         demod_power[:, [4, -4]].mean(axis=1), raw_power.sum(axis=1),
                                         *(raw_power[:, band].sum(axis=1) for band in lobe_bands))))
    trace = np.column_stack((np.concatenate(primary), np.concatenate(shape), np.concatenate(receiver)))
    controls = np.concatenate(null)
    if controls.shape[2] != phase_controls:
        raise ValueError("Wrong number of frozen phase controls")
    return trace, controls


def boot_delays(time, first, levels, selected, contrasts, offset):
    err = levels - selected
    strata = (time < ANCHOR - .06, time >= ANCHOR + .06)
    boundary = int(np.searchsorted(time, ANCHOR))
    ranges = ((0, boundary), (boundary, len(time)))
    offset_s = {}
    arrays = {}
    for width in BLOCKS:
        rng = np.random.default_rng(np.random.SeedSequence([SEED, offset, width]))
        simulated = np.empty((BOOTSTRAP, len(time), levels.shape[1]))
        calibr = np.array([np.mean(err[s], axis=0) for s in strata])
        pools = [np.flatnonzero(np.lib.stride_tricks.sliding_window_view(s, width).all(axis=1)) for s in strata]
        for side, (lo, hi) in enumerate(ranges):
            for start in range(lo, hi, width):
                stop = min(start + width, hi)
                indices = rng.choice(pools[side], BOOTSTRAP)[:, None] + np.arange(stop - start)
                simulated[:, start:stop] = selected[None, start:stop] + err[indices] - calibr[side]
        winnings = np.maximum(contrasts @ simulated, 0.)**2
        estimates = first[winnings.argmax(axis=1)]
        differences = estimates - estimates[:, :1]
        offset_s[str(width)] = np.quantile(differences, [.025, .975], axis=0).T.tolist()
        arrays[f"boot_edges_{offset}_{width}"] = estimates
    return offset_s, arrays


def score_cut(trace, times, starts, names, null, offset):
    fit = in_interval(times, FIT_INTERVAL)
    first = starts[fit]
    cand = np.flatnonzero(in_interval(first, EDGE_INTERVAL))
    step, trend, contrasts = step_bank(times[fit], first, cand)
    values = trace[fit]
    picks, fitted = step_fit(values, step, trend, contrasts)
    endpoints = first[cand[picks]]
    uncertainty, arrays = boot_delays(times[fit], first[cand], values[:, :3], fitted[:, :3], contrasts, offset)
    pre, after = (in_interval(times, r) for r in LEVEL_WINDOWS)
    ratios = 10 * np.log10(trace[pre].mean(axis=0) / trace[after].mean(axis=0))
    null_ratios = 10 * np.log10(null[pre].mean(axis=0) / null[after].mean(axis=0))
    excess = ratios[:2, None] - null_ratios
    return {
        "offset_samples": offset,
        "components": {name: {"best_drop_time_s": float(t), "before_to_after_db": float(delta)}
                       for name, t, delta in zip(names, endpoints, ratios, strict=True)},
        "phase_control_max_drop_db": null_ratios.max(axis=1).tolist(),
        "phase_controls_meeting_reference_drop": (excess <= 0).sum(axis=1).tolist(),
        "boot_delay_vs_negative_s_95pct": uncertainty,
    }, arrays


def main():
    require_run_context()
    inputs = {}

    def source(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    run = "v038-wav-shifted-lines"
    grid_meta = read(source(run, "E040", "wav_shifted_lines.json"))
    spacing = grid_meta["frequency_grid_hz"]
    phase = read(source("v039-wav-joint-phase", "E041", "wav_joint_phase.json"))
    delay = phase["timing_modes"]["0"]["0"]["2"]["polynomial_samples"]
    training = phase["groups"]["training"]
    with np.load(source(run, "E040", "wav_shifted_lines.npz"), allow_pickle=False) as z:
        short = np.array(training)[in_interval(z["centers_s"][training], (11., 13.))]
        offsets = z["offsets"][short, 0].mean(axis=0)
    with np.load(source("v039-wav-joint-phase", "E041", "wav_joint_phase.npz"), allow_pickle=False) as z:
        h = z["harmonics"]
        average = unit_rows(z["mode_0_source_0_aligned_spectra"][training].mean(axis=0))
    geom = read(source("v040-wav-tessellation", "E042", "wav_tessellation.json"))
    with np.load(source("v040-wav-tessellation", "E042", "wav_tessellation.npz"), allow_pickle=False) as z:
        if not np.array_equal(h, z["harmonic_rows"]):
            raise ValueError("E041 and E042 harmonic support differs")
        model = forward(h, geom["fitted_offset_hz"]) @ z["calibration_coefficients"]
    fitted = unit_rows(model.reshape(h.shape))
    rng = np.random.default_rng(SEED)
    controls = average[:, None, :] * np.exp(2j * np.pi * rng.random((2, SHIFT_CONTROLS, h.shape[-1])))
    templates = np.concatenate((average[:, None], fitted[:, None], controls), axis=1)
    track = ROOT / "experiments/E001/results/refined_track.npz"
    for path in (RAW_WAV, track):
        inputs[path] = sha256(path)
        if inputs[path] != grid_meta["inputs_sha256"][path.relative_to(ROOT).as_posix()]:
            raise ValueError(f"Frozen E040 WAV or track hash differs: {path.name}")
    with np.load(track, allow_pickle=False) as z:
        cw = z["quadratic_coefficients"]
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    if fs != 2_400_000 or raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype != np.int16:
        raise ValueError("Expected original 2.4 MS/s CI16 WAV")
    labels = (*PRIMARY, "negative_four_carrier", "positive_four_carrier",
              "receiver_center", "fixed_13khz", "cw_off_tone", "raw_fullband", "raw_negative_lobe", "raw_positive_lobe")
    frame = np.arange(int(WAV_INTERVAL[0] * spacing), int(WAV_INTERVAL[1] * spacing))
    arrays, outcomes = {}, []
    for offset in OFFSETS:
        coordinate = frame * PERIOD + geom["epoch_samples"] + offset
        guess = coordinate / (PERIOD * spacing)
        indices = np.rint((coordinate + np.polyval(delay, guess)) * fs / (PERIOD * spacing)).astype(int)
        ts = (indices + PERIOD / 2) / fs
        obs, null = inspect_profiles(raw, fs, indices, spacing, delay, cw, offsets, h, templates, SHIFT_CONTROLS)
        result, more = score_cut(obs, ts, indices / fs, labels, null, offset)
        arrays.update(more)
        arrays.update({f"first_{offset}": indices, f"power_{offset}": obs, f"phase_controls_power_{offset}": null})
        outcomes.append(result)
        print(offset, [round(result["components"][name]["best_drop_time_s"], 7) for name in PRIMARY], flush=True)
    report = {
        "schema": "starlink-wav-frame-termination/v1", "event_count": 1, "seed": SEED,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()},
        "frame_spacing_hz": spacing, "sampling_rate": fs, "step_prior_seconds": ANCHOR,
        "empirical_template_calibration_blocks": training, "offset_calibration_blocks": short.tolist(),
        "offsets_hz": offsets.tolist(), "series_labels": labels, "phase_control_count": SHIFT_CONTROLS,
        "fit_interval_s": FIT_INTERVAL, "edge_candidate_interval_s": EDGE_INTERVAL, "level_windows_s": LEVEL_WINDOWS,
        "boot_delays_group_frames": BLOCKS, "boot_repetitions_per_group": BOOTSTRAP, "results": outcomes,
    }
    out = experiment_output("E090")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "wav_frame_termination.npz", **arrays)
    (out / "wav_frame_termination.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
