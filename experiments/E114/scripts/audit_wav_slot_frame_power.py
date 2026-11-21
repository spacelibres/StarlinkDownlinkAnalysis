"""Test whether the near-zero class-34 average hides frame-to-frame reversals."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, RAW_WAV, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_reacquisition import sample_window
from audit_wav_slot_case34 import GROUP_WINDOWS, smooth_columns
from audit_wav_tessellation import CARRIERS, FRAME, GRID, forward


SEED = 20261121116
GROUP_FRAMES = 8
CONTROL_SHIFTS = (8, 16, 24, 32, 40, 48, 56)
DEGREES = (0, 1)
RESIDUAL_CASES = (4, 5, 12)
AMPS = (.5, 1.)
BOOTSTRAP = 499
SIGN_FLOOR = .35


def extract_frames(samples, times, fs, spacing, offset, delay, harmonics):
    coordinate = times * spacing - delay / FRAME
    ordinal = np.arange(np.floor(coordinate[0]), np.ceil(coordinate[-1]) + 1, dtype=int)
    starts = np.rint((ordinal - coordinate[0]) * fs / spacing).astype(int)
    good = (starts >= 0) & (starts + FRAME <= len(samples))
    starts, ordinal = starts[good], ordinal[good]
    index = starts[:, None] + np.arange(FRAME)
    data = samples[index] * np.exp(-2j * np.pi * offset * times[index])
    spectrum = fft.fft(data, axis=1)[:, harmonics] / FRAME
    fraction = coordinate[starts] - ordinal
    return ordinal, spectrum * np.exp(-2j * np.pi * fraction[:, None] * harmonics)


def paired_moments(a, b):
    ac, bc = a - a.mean(), b - b.mean()
    return {'coherent_real': [float(a.mean().real), float(b.mean().real)],
            'coherent_imag': [float(a.mean().imag), float(b.mean().imag)],
            'centered_real_power': float(np.mean(ac.real * bc.real)),
            'centered_quadrature_power': float(np.mean(ac.imag * bc.imag)),
            'all_complex_power': float(np.mean(a * b.conj()).real),
            'centered_frame_shift_controls': [float(np.mean(ac.real * np.roll(bc.real, step))) for step in CONTROL_SHIFTS]}


def stable_signs(projections):
    return np.all(projections >= SIGN_FLOOR, axis=0).astype(np.int8) - np.all(projections <= -SIGN_FLOOR, axis=0)


def donor_phase(spectrum, bank, donor, carrier):
    model = bank[donor][:, np.arange(bank.shape[1]) != carrier]
    amplitude = np.linalg.lstsq(model, spectrum[:, donor].mean(axis=0), rcond=1e-8)[0]
    prediction = model @ amplitude
    reference = spectrum[:, donor] @ prediction.conj()
    return reference / np.maximum(abs(reference), 1e-30)


def solve_amplitudes(spectrum, bank, cases, masks, scale, carrier, degree, split):
    """Estimate the candidate coefficient on disjoint frequency bins. Normalize by
    carrier gain from different groups of whole frames in the same window.
    """
    target_columns = cases[:, -1, None] * scale[:, carrier, None]**np.arange(degree + 1)
    base = smooth_columns(bank, scale, degree)
    design = np.column_stack((base, cases[:, :-1], target_columns))
    at_carrier = carrier * (degree + 1)
    result = []
    for mask in masks:
        gain = spectrum[:, mask] @ np.linalg.pinv(design[mask], rcond=1e-8).T
        carrier_gain = gain[:, at_carrier]
        normalizer = np.where(split, carrier_gain[~split].mean(), carrier_gain[split].mean())
        if np.any(abs(normalizer) < 1e-20):
            raise ValueError('No reference carrier gain available')
        result.append(gain[:, -(degree + 1)] / normalizer)
    return result


def case_geometry(bank, atoms, carrier, slot):
    columns = sorted(set(RESIDUAL_CASES) | {slot})
    treated = bank.copy()
    treated[:, carrier] -= atoms[:, columns].sum(axis=1)
    nuisance = [i for i in columns if i != slot]
    return treated, atoms[:, [*nuisance, slot]]


def classify_frames(spectrum, labels, bank, atoms, masks, degrees, scale, carrier, split, slot, retain=False):
    design, columns = case_geometry(bank, atoms, carrier, slot)
    fits = []
    for degree in degrees:
        first, second = solve_amplitudes(spectrum, design, columns, masks, scale, carrier, degree, split)
        score = paired_moments(first, second)
        if labels is not None:
            score['known_sign_projection_by_arm'] = [float(np.mean(a.real * labels)) for a in (first, second)]
        if retain:
            score['real_projection_by_arm'] = [first.real.tolist(), second.real.tolist()]
        fits.append(score)
    return fits


def reference_gain_for_injection(spectrum, bank, atoms, masks, carrier, slot):
    model, extras = case_geometry(bank, atoms, carrier, slot)
    fit = masks[0] | masks[1]
    both = np.column_stack((model, extras))
    return np.linalg.lstsq(both[fit], spectrum[:, fit].mean(axis=0), rcond=1e-8)[0][carrier]


def summarize(records, slot):
    outcomes = []
    rng = np.random.default_rng(SEED + 1)
    for index, degree in enumerate(DEGREES):
        groups = []
        for group in sorted({x['time_group'] for x in records}):
            cases = [x for x in records if x['time_group'] == group]
            groups.append(np.mean([x['targets'][str(slot)][index]['centered_real_power'] for x in cases]))
        groups = np.asarray(groups)
        simulated = groups[rng.integers(len(groups), size=(BOOTSTRAP, len(groups)))].mean(axis=1)
        slot_results = {q: {metric: float(np.mean([x['targets'][q][index][metric] for x in records]))
                            for metric in ('centered_real_power', 'centered_quadrature_power', 'all_complex_power')}
                        for q in records[0]['targets']}
        tests = []
        for amplitude in AMPS:
            values = np.asarray([x['injections'][str(amplitude)][index]['centered_real_power'] -
                                 x['targets'][str(slot)][index]['centered_real_power'] for x in records])
            tests.append({'added_amplitude': amplitude, 'mean_centered_power_increase': float(values.mean()),
                          'increase_quantiles_10_50_90': np.quantile(values, [.1, .5, .9]).tolist()})
        outcomes.append({'degree': degree, 'slot_moments_by_average_window': slot_results,
                         'target_group_averaged_centered_power': float(groups.mean()),
                         'bootstrap_target_centered_power_95pct': np.quantile(simulated, [.025, .975]).tolist(),
                         'synthetic_alternations': tests})
    return outcomes


def summarize_signs(records):
    counts = {str(i): Counter() for i in RESIDUAL_CASES}
    patterns = Counter()
    for row in records:
        trace = np.array([row['retained_signs'][str(i)] for i in RESIDUAL_CASES])
        for key, values in zip(counts, trace, strict=True):
            counts[key].update(values.tolist())
        for pattern in trace.T:
            if np.all(pattern != 0):
                patterns[''.join('+' if x > 0 else '-' for x in pattern)] += 1
    return {'floor': SIGN_FLOOR, 'slot_order': list(RESIDUAL_CASES), 'counts_by_slot': counts,
            'joint_observed_patterns': dict(sorted(patterns.items()))}


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    active, again = [doc('v111-wav-slot-case34' + suffix, 'E113', 'wav_slot_case34.json')
                     for suffix in ('-final', '-confirm-clean')]
    if active != again:
        raise ValueError('E113 timing receipts differ')
    grid = doc('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json')
    phase = doc('v039-wav-joint-phase', 'E041', 'wav_joint_phase.json')['timing_modes']['0']['0']['2']['polynomial_samples']
    template = doc('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    word = np.asarray(doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')['cycle_folds'][0]['signed_template'])
    resources = np.asarray(template['reference_comparison']['resource_index_by_lane_and_slot_class'])
    carrier, slot = map(int, np.argwhere(resources == active['target_resource_mod_60'])[0])
    symbols = word[(resources - active['fixed_word_phase_subtracted']) % len(word)]
    with np.load(artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.npz'), allow_pickle=False) as data:
        h = data['harmonic_rows']
        held = data['held_harmonic_mask']
    offset = template['fitted_offset_hz']
    basis = forward(h, offset).reshape(-1, len(CARRIERS), symbols.shape[1])
    bank = np.einsum('ncs,cs->nc', basis, symbols)
    harmonics = h.ravel()
    f = harmonics * 750.
    centers = CARRIERS * GRID + offset
    scale = np.clip((f[:, None] - centers) / 200_000., -2., 2.)
    donor = (harmonics < 0) | (abs(f - centers[-1]) < GRID / 3)
    region = abs(f - centers[carrier]) < GRID / 2
    frequencies = region & ~held, region & held
    atoms = basis[:, carrier] * symbols[carrier]
    track = PROJECT_ROOT / 'experiments/E001/results/refined_track.npz'
    for path in (RAW_WAV, track):
        hashes[path] = sha256(path)
        if hashes[path] != grid['inputs_sha256'][path.relative_to(PROJECT_ROOT).as_posix()]:
            raise ValueError(f'Frozen source changed: {path.name}')
    with np.load(track, allow_pickle=False) as z:
        cw = z['quadratic_coefficients']
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    result = []
    rng = np.random.default_rng(SEED)
    for index, window in enumerate(active['windows']):
        receiver = window['arms'][0]['alignment']
        if not receiver['qualified']:
            continue
        data, t, rate = sample_window(raw, fs, window['start_s'], cw, grid['frequency_grid_hz'], phase)
        ordinals, spectra = extract_frames(data, t, fs, rate, receiver['offset_hz'], receiver['delay_samples'], harmonics)
        common = donor_phase(spectra, bank, donor, carrier)
        selected = spectra * common.conj()[:, None]
        groups = np.arange(len(ordinals)) // GROUP_FRAMES
        all_groups = np.unique(groups)
        first = rng.permutation(all_groups)[:len(all_groups) // 2]
        split = np.isin(groups, first)
        measured = {str(q): classify_frames(selected, None, bank, atoms, frequencies, DEGREES, scale, carrier,
                                             split, q, retain=q in RESIDUAL_CASES)
                    for q in range(symbols.shape[1])}
        signs_by_slot = {str(q): stable_signs(np.concatenate([a['real_projection_by_arm'] for a in measured[str(q)]])).tolist()
                         for q in RESIDUAL_CASES}
        injection_gain = reference_gain_for_injection(selected, bank, atoms, frequencies, carrier, slot)
        signs = (-1.)**(ordinals % 2)
        injected = {}
        for level in AMPS:
            alternate = selected + (signs * level * injection_gain)[:, None] * atoms[:, slot]
            injected[str(level)] = classify_frames(alternate, signs, bank, atoms, frequencies, DEGREES, scale, carrier, split, slot)
        row = {'window_start_s': window['start_s'], 'time_group': index // GROUP_WINDOWS,
               'ordinals_in_window': ordinals.tolist(), 'calibration_frame_groups': first.tolist(),
               'frequency_bins_by_partition': [int(m.sum()) for m in frequencies], 'targets': measured,
               'retained_signs': signs_by_slot, 'injections': injected}
        result.append(row)
        if index % 8 == 0:
            print(window['start_s'], len(ordinals), [round(x['centered_real_power'], 5) for x in measured[str(slot)]], flush=True)
    output = {'schema': 'starlink-wav-slot-frame-power/v1', 'target_slot': slot, 'target_resource_mod_60': active['target_resource_mod_60'],
              'seed': SEED, 'inputs_sha256': {p.as_posix(): digest for p, digest in hashes.items()},
              'summary': summarize(result, slot), 'observed_signs': summarize_signs(result), 'windows': result}
    destination = experiment_output('E114')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'wav_slot_frame_power.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(output['summary'], flush=True)


if __name__ == '__main__':
    main()
