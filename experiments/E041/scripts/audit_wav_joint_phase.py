"""Test cross-band timing and distinguish public template hypotheses."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_shifted_lines import HARMONICS, PERIOD


SEED = 2026112141
RATE = 2400000
DELAY = np.arange(-240, 241) * .05
LS = 1056
WIDE_FRAME = 320000
TONE_OFFSETS = 117187.5 * np.arange(-4, 5)
PHASE_CONTROLS = 399
TEMPLATE_CONTROLS = 99


def unit(x):
    return x / np.maximum(abs(x), 1e-30)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)


def phase_model(x, inferred, shift, train, held, post, permutations):
    kernel = np.exp(2j*np.pi*shift[:, None, None]*HARMONICS / PERIOD)
    corrected = x * kernel
    z = np.sum(corrected * inferred.conj(), axis=2)
    carrier_phase = unit(z)
    relative = carrier_phase[:, 1] * carrier_phase[:, 0].conj()
    fixed = unit(relative[train].mean())
    comparison = (relative * fixed.conj()).real
    other = carrier_phase[permutations, 1]
    held_negative = carrier_phase[held, 0]
    null = (other * held_negative.conj() * fixed.conj()).real.mean(axis=1)
    aligned = corrected * carrier_phase[:, 0, None, None].conj()
    return {"held_signed_phase": float(comparison[held].mean()), "held_median_phase": float(np.median(comparison[held])),
            "training_signed_phase": float(comparison[train].mean()), "post_signed_phase": float(comparison[post].mean()),
            "held_permutation_95pct": float(np.quantile(null, .95)),
            "permutations_ge_measured": int(np.count_nonzero(null >= comparison[held].mean())),
            "held_band_shape": np.median(abs(z[held]), axis=0).tolist(),
            "calibration_phase_angle": float(np.angle(fixed))}, aligned, relative


def estimate_timing(waves, times, offsets, train, held, post, swaps):
    spectra = fft.fft(waves, axis=-1)[:, HARMONICS]
    x = normalize(spectra)
    inferred = np.array([np.linalg.svd(x[train, band], full_matrices=False)[2][0] for band in (0, 1)])
    delay_rotor = np.exp(2j*np.pi*HARMONICS[:, :, None]*DELAY / PERIOD)
    time_fit = np.stack([DELAY[abs((x[:, band]*inferred[band].conj()) @ delay_rotor[band]).argmax(axis=1)]
                         for band in (0, 1)], axis=1)
    centroid = np.sum(abs(inferred)**2 * HARMONICS, axis=1)
    difference = offsets[:, 1] - offsets[:, 0]
    scores = {}
    output = {}
    for source in (0, 1):
        rows = {}
        for degree in (0, 1, 2):
            polynomial = np.polyfit(times[train], time_fit[train, source], degree)
            pred = np.polyval(polynomial, times)
            metrics, aligned, relative = phase_model(x, inferred, pred, train, held, post, swaps)
            expected_f = -np.polyval(np.polyder(polynomial), times)*(centroid[1]-centroid[0])/PERIOD
            if degree == 2:
                correlation = float(np.corrcoef(expected_f[held], difference[held])[0, 1])
                output[source] = (pred, aligned, relative, expected_f)
            else:
                correlation = None
            timing_error = time_fit[held, 1-source] - pred[held]
            rows[str(degree)] = {**metrics, "polynomial_samples": polynomial.tolist(),
                                 "held_delay_rms_samples_other_band": float(np.sqrt(np.mean(timing_error**2))),
                                 "held_frequency_difference_rms_hz": float(np.sqrt(np.mean((expected_f[held]-difference[held])**2))),
                                 "held_frequency_difference_correlation": correlation}
        scores[str(source)] = rows
    return scores, time_fit, output


def published_templates(sss, rotations):
    """Reconstruct public fields; MATLAB array column 0 corresponds to slot 1."""
    register = [0, 0, 1, 1, 0, 1, 0]
    sequence = []
    for _ in range(127):
        tail = register[2] ^ register[6]
        register = [tail, *register[:-1]]
        sequence.append(tail)
    portion = np.exp(-1j*np.pi/4 - .5j*np.pi*np.cumsum(2*np.r_[0, sequence[::-1]]-1))
    pss = np.r_[-portion, np.tile(portion, 7)]
    pss = np.r_[-pss[-32:], pss]
    def symbols(frequency):
        content = 32*fft.ifft(frequency, axis=0)
        return np.concatenate((content[-32:], content), axis=0).T.ravel()
    sss = symbols(sss.reshape(-1, 1))
    body = (1j)**rotations[:, 1:].astype(int)
    body[np.r_[0, 1, 1022, 1023, np.arange(488, 496), np.arange(528, 536)]] = 0
    body = symbols(body)
    definitions = {
        "pss": (0, pss),
        "sss": (LS, sss),
        "pss_sss": (0, np.r_[pss, sss]),
        "qin_body": (2*LS, body),
        "combined": (0, np.r_[pss, sss, body]),
    }
    time = np.arange(WIDE_FRAME) / 240e6
    bank = {}
    for k, (first, waveform) in definitions.items():
        variants = []
        for conjugate in (False, True):
            signal = waveform.conj() if conjugate else waveform
            for offset in TONE_OFFSETS:
                wave = np.zeros(WIDE_FRAME, dtype=complex)
                extent = slice(first, first+len(signal))
                wave[extent] = signal * np.exp(2j*np.pi*offset*time[extent])
                variants.append((float(offset), conjugate, normalize(fft.fft(wave)[HARMONICS])))
        bank[k] = variants
    return bank


def fit_bank(templates, calibration, control_phase):
    score = np.full(len(control_phase), -1.)
    choice = np.full((len(control_phase), 2), -1, dtype=int)
    for j, (_, _, spectrum) in enumerate(templates):
        proposed = spectrum[None] * control_phase
        terms = np.zeros((len(control_phase), 2, PERIOD), dtype=complex)
        for side in (0, 1):
            terms[:, side, HARMONICS[side]] = proposed[:, side].conj() * calibration[side]
        match = (abs(fft.ifft(terms, axis=2))**2).mean(axis=1) * PERIOD**2
        shift = match.argmax(axis=1)
        merit = match[np.arange(len(control_phase)), shift]
        better = merit > score
        choice[better, 0] = j
        choice[better, 1] = shift[better]
        score[better] = merit[better]
    return choice, score


def template_test(bank, calibration, held, post, phases):
    checks = {}
    for name, templates in bank.items():
        choices, scores = fit_bank(templates, calibration, phases)
        evaluated = []
        post_match = None
        for i, (j, lag) in enumerate(choices):
            a = templates[j][2] * phases[i] * np.exp(-2j*np.pi*HARMONICS*lag/PERIOD)
            evaluated.append(abs(np.sum(held * a.conj(), axis=1))**2)
            if i == 0:
                post_match = (abs(np.sum(post * a.conj(), axis=1))**2).tolist()
        x = np.array(evaluated)
        chosen, lag = choices[0]
        checks[name] = {"conjugated": templates[chosen][1], "carrier_shift_hz": templates[chosen][0],
                        "delay_samples": int(lag), "calibration_projection_power": float(scores[0]),
                        "held_projection_power_by_band": x[0].tolist(), "post_projection_power_by_band": post_match,
                        "controls_ge_observed": int(np.count_nonzero(x[1:].mean(axis=1) >= x[0].mean())),
                        "control_held_mean_power": x[1:].mean(axis=1).tolist()}
    return checks


def window_power(template):
    n = 22  # 8.8 us at 2.4 MS/s rounded up.
    wave = np.zeros((2, PERIOD), dtype=complex)
    for side in (0, 1):
        wave[side, HARMONICS[side]] = template[side]
    power = abs(fft.ifft(wave, axis=1))**2
    windows = np.lib.stride_tricks.sliding_window_view(np.c_[power, power[:, :n-1]], n, axis=1)
    ratios = windows.sum(axis=2)/power.sum(axis=1, keepdims=True)
    return ratios.max(axis=1).tolist(), power


def main():
    require_run_context()
    used = {}
    pair = "v038-wav-shifted-lines"
    meta = read(checked_artifact(pair, "E040", "wav_shifted_lines.json", used))
    measured = checked_artifact(pair, "E040", "wav_shifted_lines.npz", used)
    sss = ROOT / "references/qin/sssVec.mat"
    qin = ROOT / "references/qin/referenceTemplate.mat"
    for path in (sss, qin):
        used[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    train, held, post = (np.asarray(meta[k]) for k in ("calibration_blocks", "heldout_blocks", "post_blocks"))
    rng = np.random.default_rng(SEED)
    swaps = np.array([rng.permutation(held) for _ in range(PHASE_CONTROLS)])
    with np.load(measured, allow_pickle=False) as samples:
        t = samples["centers_s"]
        method = {}
        arrays = {}
        for mode in (0, 1):
            models, observed, fitted = estimate_timing(samples["folded"][:, 0, mode].astype(complex),
                                                       t, samples["offsets"][:, 0], train, held, post, swaps)
            method[str(mode)] = models
            arrays[f"mode_{mode}_measured_delay"] = observed
            for side, (d, a, r, fd) in fitted.items():
                for label, value in (("delay", d), ("aligned_spectra", a), ("relative_phase", r), ("frequency_difference", fd)):
                    arrays[f"mode_{mode}_source_{side}_{label}"] = value
        aligned = arrays["mode_0_source_0_aligned_spectra"]
    cal, reserve, after = (normalize(aligned[k].mean(axis=0)) for k in (train, held, post))
    published = published_templates(loadmat(sss)["sssVecFull"], loadmat(qin)["referenceTemplateRotations"])
    randomized = np.exp(2j*np.pi*rng.random((TEMPLATE_CONTROLS, *cal.shape)))
    phases = np.concatenate((np.ones_like(cal)[None], randomized), axis=0)
    known_index, known_shift = 6, 177
    synthetic = published["qin_body"][known_index][2]*np.exp(-2j*np.pi*HARMONICS*known_shift/PERIOD)
    synth_choice, synth_score = fit_bank(published["qin_body"], synthetic, phases[:1])
    if synth_choice[0].tolist() != [known_index, known_shift] or synth_score[0] < .999999:
        raise ValueError("Synthetic public reference could not be recovered")
    tests = template_test(published, cal, reserve, after, phases)
    concentration, power = window_power(reserve)
    report = {"schema": "starlink-wav-joint-phase/v1", "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p,d in used.items()},
              "seed": SEED, "phase_controls": PHASE_CONTROLS, "template_controls": TEMPLATE_CONTROLS,
              "sample_rate_hz": RATE, "groups": {"training": train.tolist(), "held": held.tolist(), "post": post.tolist()},
              "timing_modes": method, "template_tests": tests, "template_offsets_hz": TONE_OFFSETS.tolist(),
              "synthetic_public_projection_power": float(synth_score[0]), "synthetic_template_index": int(synth_choice[0, 0]),
              "synthetic_delay_samples": int(synth_choice[0, 1]),
              "max_22_sample_window_power_fraction": concentration,
              "scope": "Relative phase and clock transfer within one WAV. Known waveforms tested with constant complex gain per band."}
    out = experiment_output("E041")
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav_joint_phase.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out / "wav_joint_phase.npz", **arrays, centers_s=t, harmonics=HARMONICS,
                        reserve_spectrum=reserve, folded_power=power, phase_swaps=swaps)
    print("Held-out joint phase", method["0"]["0"]["2"]["held_signed_phase"], flush=True)


if __name__ == "__main__":
    main()
