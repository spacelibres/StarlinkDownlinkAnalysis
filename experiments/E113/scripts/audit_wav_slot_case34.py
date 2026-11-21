"""Distinguish a missing, flipped or reference-aligned slot-class contribution."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, RAW_WAV, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_reacquisition import (
    OFFSETS, complex_tone_grid, held_score, injected_wave, profile_delay, sample_window, strongest_offsets,
)
from audit_wav_tessellation import CARRIERS, FRAME, GRID, forward


RESOURCE = 34
DEGREES = (0, 1, 2)
HYPOTHESES = (1, 0, -1)
STEPS_PER_LAG = 8
MIN_DONOR_GAIN = .05
PRE_WINDOWS = tuple(round(.32 * index, 2) for index in range(43))
SEED = 20261121114
GROUP_WINDOWS = 4
BOOTSTRAP = 499


def smooth_columns(bank, scales, degree):
    return (bank[:, :, None] * scales[:, :, None]**np.arange(degree + 1)).reshape(len(bank), -1)


def compare_amplitudes(y, without, slot, carrier, scales, degree, fit, held):
    base = smooth_columns(without, scales, degree)
    target = slot[:, None] * scales[:, carrier, None]**np.arange(degree + 1)
    positions = slice(carrier * (degree + 1), (carrier + 1) * (degree + 1))
    power = np.vdot(y[held], y[held]).real

    def evaluate(design):
        gains = np.linalg.lstsq(design[fit], y[fit], rcond=1e-8)[0]
        residual = y[held] - design[held] @ gains
        return float(np.vdot(residual, residual).real / power), gains

    scores = []
    for coefficient in HYPOTHESES:
        change = base.copy()
        change[:, positions] += coefficient * target
        score, _ = evaluate(change)
        scores.append(score)
    free_error, gains = evaluate(np.column_stack((base, target)))
    nominal = target[held] @ gains[positions]
    free = target[held] @ gains[-(degree + 1):]
    ratio = np.vdot(nominal, free) / max(np.vdot(nominal, nominal).real, 1e-30)
    return {'winner': HYPOTHESES[int(np.argmin(scores))], 'held_nmse_reference_zero_flip': scores,
            'free_coefficient': [float(ratio.real), float(ratio.imag)], 'free_held_nmse': free_error}


def donor_alignment(spectra, bank, harmonics, donor_fit, donor_held, known_offset):
    trials = []
    for i in strongest_offsets(spectra, donor_fit, known_offset):
        y = spectra[i]
        _, lag, gains, power = profile_delay(bank[None, donor_fit], y[donor_fit], harmonics[donor_fit], FRAME * STEPS_PER_LAG)
        trials.append((power, i, lag / STEPS_PER_LAG, gains))
    _, i, lag, gains = max(trials, key=lambda x: x[0])
    _, held = held_score(bank, gains, spectra[i], lag, harmonics, ~donor_held)
    aligned = spectra[i] * np.exp(2j * np.pi * harmonics * lag / FRAME)
    return aligned, {'offset_hz': float(OFFSETS[i]), 'delay_samples': lag,
                     'donor_held_error_reduction': held, 'qualified': held > MIN_DONOR_GAIN}


def end_to_end(spectra, masks, bank, basis, symbols, harmonics, scales, carrier, usual, target_case, other_cases):
    donor, target, held = masks
    raw = []
    for mask in (~held, held):
        y, alignment = donor_alignment(spectra, bank[:, np.arange(len(CARRIERS)) != carrier], harmonics,
                                       mask & donor, ~mask & donor, usual)
        paths = []
        for degree in DEGREES:
            def test(slot):
                blank = bank.copy()
                component = basis[:, carrier, slot] * symbols[carrier, slot]
                blank[:, carrier] -= component
                return compare_amplitudes(y, blank, component, carrier, scales, degree, mask, ~mask & target)

            controls = {slot: test(slot)['winner'] for slot in other_cases} if degree in (0, 2) else {}
            paths.append({'degree': degree, 'target': test(target_case), 'other_class_winners': controls})
        raw.append({'alignment': alignment, 'paths': paths})
    return raw


def summarize(windows):
    results = []
    rng = np.random.default_rng(SEED + 1)
    qualified = [row for row in windows if all(arm['alignment']['qualified'] for arm in row['arms'])]
    for index, degree in enumerate(DEGREES):
        evidence = [a['paths'][index]['target'] for x in qualified for a in x['arms']]
        groups = []
        for group in sorted({row['time_group'] for row in qualified}):
            blocks = [a['paths'][index]['target']['held_nmse_reference_zero_flip']
                      for row in qualified if row['time_group'] == group for a in row['arms']]
            groups.append(np.mean(blocks, axis=0))
        grouped = np.stack(groups)
        sampled = grouped[rng.integers(len(groups), size=(BOOTSTRAP, len(groups)))].mean(axis=1)
        use = [a['paths'][index]['other_class_winners'] for x in qualified for a in x['arms']]
        results.append({'degree': degree, 'qualified_windows': len(qualified), 'groups': len(groups),
                        'group_means_nmse_reference_zero_flip': grouped.mean(axis=0).tolist(),
                        'group_bootstrap_delta_bounds_zero_minus_reference_and_flip': np.quantile(
                            sampled[:, 1, None] - sampled[:, [0, 2]], [.025, .975], axis=0).tolist(),
                        'winners_reference_zero_flip': [sum(x['winner'] == k for x in evidence) for k in HYPOTHESES],
                        'free_coefficient_mean_complex': np.mean([x['free_coefficient'] for x in evidence], axis=0).tolist(),
                        'free_inside_half_amplitude': int(np.count_nonzero(
                            np.linalg.norm([x['free_coefficient'] for x in evidence], axis=1) < .5)),
                        'other_classes_reference_winners': sum(sum(v == 1 for v in x.values()) for x in use),
                        'other_classes_cases': sum(len(x) for x in use)})
    return results


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    grid = doc('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json')
    delay = doc('v039-wav-joint-phase', 'E041', 'wav_joint_phase.json')['timing_modes']['0']['0']['2']['polynomial_samples']
    history = doc('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    word = np.asarray(doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')['cycle_folds'][0]['signed_template'])
    usual = np.mean(doc('v088-wav-frame-cut-stratified', 'E090', 'wav_frame_termination.json')['offsets_hz'])
    reference = doc('v110-wav-cycle-reacquisition-final', 'E112', 'wav_cycle_reacquisition.json')['wav'][1]['arms'][0]
    phase = history['reference_comparison']['all_phases'][0]['phase_subtracted']
    resource = np.asarray(history['reference_comparison']['resource_index_by_lane_and_slot_class'])
    carrier, slot = np.argwhere(resource == RESOURCE)[0]
    symbols = word[(resource - phase) % len(word)]
    with np.load(artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.npz'), allow_pickle=False) as z:
        h = z['harmonic_rows']
        held = z['held_harmonic_mask']
    bases = forward(h, history['fitted_offset_hz']).reshape(-1, 4, 15)
    harmonics = h.ravel()
    bank = np.einsum('ncs,cs->nc', bases, symbols)
    frequ = harmonics * 750.
    means = CARRIERS * GRID + history['fitted_offset_hz']
    target = abs(frequ - means[carrier]) < GRID / 2
    donor = (harmonics < 0) | (abs(frequ - means[-1]) < GRID / 3)
    scales = np.clip((frequ[:, None] - means) / 200_000., -2., 2.)
    masks = donor, target, held
    other = [i for i in range(bases.shape[-1]) if i != slot]
    track = PROJECT_ROOT / 'experiments/E001/results/refined_track.npz'
    for path in (RAW_WAV, track):
        hashes[path] = sha256(path)
        if hashes[path] != grid['inputs_sha256'][path.relative_to(PROJECT_ROOT).as_posix()]:
            raise ValueError(f'Frozen source changed: {path.name}')
    with np.load(track, allow_pickle=False) as z:
        cw = z['quadratic_coefficients']
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    records = []
    for index, start in enumerate(PRE_WINDOWS):
        y, _, spacing = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        spectra = complex_tone_grid(y, fs, OFFSETS[:, None] + harmonics * spacing)
        arms = end_to_end(spectra, masks, bank, bases, symbols, harmonics, scales, carrier, usual, slot, other)
        records.append({'start_s': start, 'time_group': index // GROUP_WINDOWS, 'arms': arms})
        if index % 8 == 0:
            print(start, [x['paths'][0]['target']['winner'] for x in arms], flush=True)
    controls = []
    rng = np.random.default_rng(SEED)
    gains = np.array([complex(*pair) for pair in reference['channel_gains']])
    for start in (13.88, 14.48, 15.48):
        y, time, spacing = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        offset = float(rng.choice(OFFSETS[8:-8]))
        lag = int(rng.integers(FRAME))
        freq = OFFSETS[:, None] + harmonics * spacing
        for coefficient in HYPOTHESES:
            local = bank.copy()
            local[:, carrier] += (coefficient - 1) * symbols[carrier, slot] * bases[:, carrier, slot]
            z = injected_wave(local, harmonics, gains, time, spacing, offset, lag)
            spectra = complex_tone_grid(y + z, fs, freq)
            arms = end_to_end(spectra, masks, bank, bases, symbols, harmonics, scales, carrier, usual, slot, [])
            controls.append({'start_s': start, 'injected_target_coefficient': coefficient, 'injected_offset_hz': offset,
                             'injected_delay': lag, 'arms': arms})
    result = {'schema': 'starlink-wav-slot-case34/v1', 'seed': SEED, 'target_resource_mod_60': RESOURCE,
              'target_carrier': int(CARRIERS[carrier]), 'target_slot_mod_15': int(slot), 'fixed_word_phase_subtracted': phase,
              'target_band_hz': [float(frequ[target].min()), float(frequ[target].max())],
              'inputs_sha256': {path.as_posix(): digest for path, digest in hashes.items()},
              'summary': summarize(records), 'windows': records, 'injected_controls': controls}
    out = experiment_output('E113')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wav_slot_case34.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
