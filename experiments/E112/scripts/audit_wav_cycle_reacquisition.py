"""Test whether the original WAV repeated structure survives with a new phase."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import wavfile
from scipy.signal.windows import hann

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, RAW_WAV, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_transfer import raw_phase
from audit_wav_tessellation import FRAME, forward


SEED = 20261121113
SECONDS = .16
PRE = (12.48, 12.80, 13.12, 13.44)
POST = (13.68, 13.88, 14.08, 14.28, 14.48, 14.68, 14.88, 15.08, 15.28, 15.48)
OFFSETS = np.arange(-375., 376., 5.)
EXTRA_OFFSETS = 3
INJECTION_FACTORS = (1., .5, .25)


def complex_tone_grid(signal, fs, frequencies):
    """Fourier coefficients with a centered time axis and unit tone gain."""
    window = hann(len(signal), sym=False)
    nfft = fft.next_fast_len(4 * len(signal))
    f = fft.fftshift(fft.fftfreq(nfft, 1 / fs))
    raw = fft.fftshift(fft.fft(signal * window, n=nfft)) / window.sum()
    centered = raw * np.exp(2j * np.pi * f * (len(signal) - 1) / (2 * fs))
    return np.interp(frequencies, f, centered.real) + 1j * np.interp(frequencies, f, centered.imag)


def profile_delay(bank, y, harmonic_indices, length=FRAME):
    """Profile linear gains and maximize over each waveform and integer lag."""
    normals = np.einsum('fnc,fnd->fcd', bank.conj(), bank)
    inverse = np.linalg.pinv(normals, hermitian=True, rcond=1e-10)
    terms = np.zeros((len(bank), bank.shape[-1], length), dtype=complex)
    terms[:, :, harmonic_indices] = (bank.conj() * y[None, :, None]).transpose(0, 2, 1)
    rhs = fft.ifft(terms, axis=2) * length
    gains = np.einsum('fcd,fdt->fct', inverse, rhs)
    objective = np.einsum('fct,fct->ft', rhs.conj(), gains).real
    phase, delay = np.unravel_index(objective.argmax(), objective.shape)
    return int(phase), int(delay), gains[phase, :, delay], float(objective[phase, delay])


def strongest_offsets(spectra, fit, known):
    power = abs(spectra[:, fit])**2
    relative = power / np.maximum(power.mean(axis=0), 1e-30)
    score = relative.mean(axis=1)
    chosen = [int(np.argmin(abs(OFFSETS - known)))]
    for i in np.argsort(-score, kind='stable'):
        if min(abs(OFFSETS[i] - OFFSETS[j]) for j in chosen) < 15:
            continue
        chosen.append(int(i))
        if len(chosen) == EXTRA_OFFSETS + 1:
            break
    return chosen


def held_score(bank, gains, y, delay, harmonics, fit):
    rotor = np.exp(-2j * np.pi * harmonics[~fit] * delay / FRAME)
    prediction = (bank[~fit] @ gains) * rotor
    norm = np.linalg.norm(y[~fit]) * np.linalg.norm(prediction)
    if norm < 1e-30:
        return 0., 0.
    co = float(np.vdot(prediction, y[~fit]).real / norm)
    power = np.vdot(y[~fit], y[~fit]).real
    gain = (2 * np.vdot(prediction, y[~fit]).real - np.vdot(prediction, prediction).real) / power
    return co, float(gain)


def search(spectra, bank, harmonics, fit, known_offset):
    offsets = strongest_offsets(spectra, fit, known_offset)
    trials = []
    for index in offsets:
        y = spectra[index]
        phase, delay, gains, score = profile_delay(bank[:, fit], y[fit], harmonics[fit])
        trials.append((score, index, phase, delay, gains))
    score, index, phase, delay, gains = max(trials, key=lambda item: item[0])
    y = spectra[index]
    co, reduction = held_score(bank[phase], gains, y, delay, harmonics, fit)
    sides = [held_score(bank[phase], gains, y, delay, harmonics, fit | exclude)[1]
             for exclude in (harmonics > 0, harmonics < 0)]
    offsets = [float(OFFSETS[i]) for _, i, _, _, _ in trials]
    return {'offset_candidates_hz': offsets, 'offset_hz': float(OFFSETS[index]), 'word_phase_subtracted': phase,
            'delay_samples': delay, 'calibration_projected_power': score,
            'held_signed_coherence': co, 'held_relative_error_reduction': reduction,
            'held_relative_error_reduction_negative_positive': sides,
            'channel_gains': [[float(v.real), float(v.imag)] for v in gains]}


def sample_window(raw, sample_rate, start, cw, frame_rate, delay_polynomial):
    ix = round(start * sample_rate) + np.arange(round(SECONDS * sample_rate))
    center = ix.mean() / sample_rate
    t = (ix - ix.mean()) / sample_rate
    pair = raw[ix].astype(float) / 32768
    values = (pair[:, 0] + 1j * pair[:, 1]) * np.exp(-1j * raw_phase(cw, ix / sample_rate))
    spacing = frame_rate - np.polyval(np.polyder(delay_polynomial), center) / FRAME
    return values, t, float(spacing)


def injected_wave(bank, harmonics, gain, time, spacing, offset, delay):
    upscale = 8 * FRAME
    coefficients = np.zeros(upscale, dtype=complex)
    coefficients[harmonics] = bank @ gain
    wave = fft.ifft(coefficients) * upscale
    coordinate = (time * spacing * FRAME - delay) * (upscale / FRAME)
    index = np.floor(coordinate).astype(int)
    frac = coordinate - index
    response = (1 - frac) * wave[index % upscale] + frac * wave[(index + 1) % upscale]
    return response * np.exp(2j * np.pi * offset * time)


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    grid = doc('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json')
    timing = doc('v039-wav-joint-phase', 'E041', 'wav_joint_phase.json')
    delay = timing['timing_modes']['0']['0']['2']['polynomial_samples']
    geometry = doc('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    word = np.asarray(doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')['cycle_folds'][0]['signed_template'])
    offsets = doc('v088-wav-frame-cut-stratified', 'E090', 'wav_frame_termination.json')['offsets_hz']
    known_offset = float(np.mean(offsets))
    resource = np.asarray(geometry['reference_comparison']['resource_index_by_lane_and_slot_class'])
    with np.load(artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.npz'), allow_pickle=False) as rows:
        harmonics = rows['harmonic_rows']
        test = rows['held_harmonic_mask']
    basis = forward(harmonics, geometry['fitted_offset_hz']).reshape(-1, 4, 15)
    phases = np.arange(len(word))[:, None, None]
    bank = np.einsum('ncs,pcs->pnc', basis, word[(resource[None] - phases) % len(word)])
    harmonics = harmonics.ravel()
    track = PROJECT_ROOT / 'experiments/E001/results/refined_track.npz'
    for path in (RAW_WAV, track):
        hashes[path] = sha256(path)
        if hashes[path] != grid['inputs_sha256'][path.relative_to(PROJECT_ROOT).as_posix()]:
            raise ValueError(f'Frozen source changed: {path.name}')
    with np.load(track, allow_pickle=False) as z:
        cw = z['quadratic_coefficients']
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    if fs != 2_400_000 or raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype != np.int16:
        raise ValueError('Expected the 2.4 MS/s complex WAV')
    records = []
    for start in (*PRE, *POST):
        values, _, spacing = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        spectra = complex_tone_grid(values, fs, OFFSETS[:, None] + spacing * harmonics)
        arms = [search(spectra, bank, harmonics, fit, known_offset) for fit in (~test, test)]
        records.append({'start_s': start, 'frame_spacing_hz': spacing, 'arms': arms})
        print(start, [(x['word_phase_subtracted'], x['offset_hz'], round(x['held_relative_error_reduction'], 4))
                      for x in arms], flush=True)
    reference = records[1]['arms'][0]
    reference_gains = np.asarray([complex(*value) for value in reference['channel_gains']])
    rng = np.random.default_rng(SEED)
    synthetic = []
    for start in (POST[0], POST[4], POST[-1]):
        values, time, spacing = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        phase = int(rng.integers(len(bank)))
        offset = float(rng.choice(OFFSETS[8:-8]))
        epoch = int(rng.integers(FRAME))
        model = injected_wave(bank[phase], harmonics, reference_gains, time, spacing, offset, epoch)
        freq = OFFSETS[:, None] + spacing * harmonics
        for level in INJECTION_FACTORS:
            spectra = complex_tone_grid(values + level * model, fs, freq)
            arms = [search(spectra, bank, harmonics, fit, known_offset) for fit in (~test, test)]
            synthetic.append({'start_s': start, 'injected_delay_samples': epoch, 'injected_offset_hz': offset,
                              'injected_word_phase_subtracted': phase, 'amplitude_multiplier': level, 'arms': arms})
        print('injection', start, phase, offset, epoch, flush=True)
    result = {'schema': 'starlink-wav-cycle-reacquisition/v1', 'seed': SEED, 'window_s': SECONDS,
              'grid_offsets_hz': OFFSETS.tolist(), 'reference_for_injection': records[1]['start_s'],
              'inputs_sha256': {p.as_posix(): digest for p, digest in hashes.items()},
              'wav': records, 'injection_controls': synthetic}
    out = experiment_output('E112')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wav_cycle_reacquisition.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
