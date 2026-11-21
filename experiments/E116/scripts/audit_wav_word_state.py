"""Compare shifted word and local three-slot models on held WAV harmonics."""

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, RAW_WAV, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_reacquisition import sample_window
from audit_wav_slot_case34 import GROUP_WINDOWS
from audit_wav_slot_frame_power import CONTROL_SHIFTS, DEGREES, SIGN_FLOOR, extract_frames
from audit_wav_tessellation import CARRIERS, GRID, forward


SEED = 20261121118
BOOTSTRAP = 499
POST = (13.68, 14.48, 15.48)
MODELS = ('fixed', 'three_slots', 'word_shift', 'shift_and_three_slots', 'negative_side_shift')
REGIONS = ('target', 'other', 'all')


def build_symbols(word, resources, carrier, slots):
    phases = np.arange(len(word))[:, None, None]
    shifts = word[(resources[None] - phases) % len(word)]
    variant = shifts.copy()
    variant[:, carrier, slots] *= -1
    return np.stack((shifts, variant))


@dataclass
class ProfiledBank:
    columns: np.ndarray
    calibration: np.ndarray
    inverse: np.ndarray
    orthogonal: np.ndarray

    def objectives(self, spectra):
        y = spectra[:, self.calibration].T
        projection = self.orthogonal @ y
        return np.sum(abs(projection)**2, axis=1)

    def predict(self, spectra, selected, held):
        y = spectra[:, self.calibration]
        gains = np.einsum('fcn,fn->fc', self.inverse[selected], y)
        return np.einsum('fnc,fc->fn', self.columns[selected][:, held], gains)


def fit_bank(columns, calibration):
    matrix = columns[:, calibration]
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    good = s > 1e-8*s[:, :1]
    inverse_s = np.divide(1., s, out=np.zeros_like(s), where=good)
    u_h = u.conj().swapaxes(-1, -2)
    inverse = (vh.conj().swapaxes(-1, -2)*inverse_s[:, None, :]) @ u_h
    return ProfiledBank(columns, calibration, inverse, u_h*good[:, :, None])


def score_selected(bank, spectra, selected, held):
    return np.sum(abs(spectra[:, held] - bank.predict(spectra, selected, held))**2, axis=1)


def design(basis, symbols, scale, degree):
    waveforms = np.einsum('nks,pks->pnk', basis, symbols)
    return (waveforms[:, :, :, None] * scale[None, :, :, None]**np.arange(degree + 1)).reshape(
        len(symbols), len(basis), -1)


def best_among(scores, candidates):
    candidates = np.asarray(candidates)
    allowed = scores[candidates]
    best = allowed.max(axis=0)
    tied = np.isclose(allowed, best[None], rtol=1e-10, atol=1e-24)
    return candidates[tied.argmax(axis=0)], tied.sum(axis=0)


def chosen_models(full, negative, phase, word_size):
    fixed = np.full(full.shape[1], phase)
    return [(fixed, np.ones_like(fixed)), best_among(full, [phase, phase + word_size]),
            best_among(full, np.arange(word_size)), best_among(full, np.arange(2*word_size)),
            best_among(negative, np.arange(word_size))]


def relative_symbols(symbols, fixed_phase, carrier, slots):
    fixed = symbols[0, fixed_phase, carrier]
    relation = symbols[0, :, carrier]*fixed
    others = ~np.isin(np.arange(len(fixed)), slots)
    polarity = np.sign(relation[:, others].sum(axis=1))
    return (polarity[:, None]*relation[:, slots]).astype(int)


def sign_diagnostics(spectra_signs, relative, negative_phase):
    eligible = np.abs(spectra_signs) >= SIGN_FLOOR
    hard = np.sign(spectra_signs)
    triplet = eligible.all(axis=1)
    prediction = relative[negative_phase]
    usable = eligible & (prediction != 0)
    both = triplet & (prediction != 0).all(axis=1)
    return {'individual_observed': int(usable.sum()),
            'individual_agrees': int(np.count_nonzero(usable & (hard == prediction))),
            'individual_agrees_fixed': int(np.count_nonzero(usable & (hard == 1))),
            'triplets_observed': int(both.sum()),
            'triplets_agree': int(np.count_nonzero(both & np.all(hard == prediction, axis=1))),
            'triplets_agree_fixed': int(np.count_nonzero(both & np.all(hard == 1, axis=1)))}


def evaluate_arm(spectra, bank, donor, target_mask, fixed_phase, word_size, relative, projections=None):
    fit = bank.calibration
    y = spectra[:, ~fit]
    masks = (target_mask[~fit], ~target_mask[~fit], np.ones(y.shape[1], dtype=bool))
    base = [float(np.sum(abs(y[:, mask])**2)) for mask in masks]
    full = bank.objectives(spectra)
    negative = donor.objectives(spectra)
    records = []
    for selected, tied in chosen_models(full, negative, fixed_phase, word_size):
        r = y - bank.predict(spectra, selected, ~fit)
        errors = [float(np.sum(abs(r[:, mask])**2)) for mask in masks]
        records.append({'phase': (selected % word_size).tolist(), 'triad_reversed': (selected >= word_size).tolist(),
                        'tied_calibration_candidates': tied.tolist(),
                        'relative_held_error': (np.asarray(errors)/base).tolist()})
    phase = np.array(records[-1]['phase'])
    shifted = [float(score_selected(bank, spectra, np.roll(phase, lag), ~fit & target_mask).sum()/base[0])
               for lag in CONTROL_SHIFTS]
    output = {'held_energy_by_region': base, 'models': dict(zip(MODELS, records, strict=True)),
              'negative_side_wrong_frame_target_errors': shifted}
    if projections is not None:
        output['sign_diagnostics'] = sign_diagnostics(projections, relative, phase)
    return output


def summarize(windows, rng):
    outcomes = []
    for degree in DEGREES:
        for arm in (0, 1):
            rows = [w['fits'][degree][arm] for w in windows]
            errors = np.asarray([[r['models'][name]['relative_held_error'] for name in MODELS] for r in rows])
            groups = np.array([w['time_group'] for w in windows])
            grouped = np.stack([errors[groups == g].mean(axis=0) for g in np.unique(groups)])
            delta = grouped - grouped[:, :1]
            resample = rng.integers(len(grouped), size=(BOOTSTRAP, len(grouped)))
            side = [row['sign_diagnostics'] for row in rows]
            outcome = {'degree': degree, 'arm': arm,
                       'group_mean_relative_error': grouped.mean(axis=0).tolist(),
                       'delta_to_fixed_percentile_95': np.quantile(delta[resample].mean(axis=1), [.025, .975], axis=0).tolist(),
                       'mean_negative_side_wrong_frame_target_errors': np.mean(
                           [r['negative_side_wrong_frame_target_errors'] for r in rows], axis=0).tolist(),
                       'negative_side_sign_diagnostics': {key: sum(r[key] for r in side) for key in side[0]},
                       'windows_shift_beats_triad_by_region': np.sum(errors[:, 2] < errors[:, 1], axis=0).tolist(),
                       'windows_triad_beats_shift_by_region': np.sum(errors[:, 1] < errors[:, 2], axis=0).tolist()}
            outcomes.append(outcome)
    return outcomes


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    geometry = doc('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    grid = doc('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json')
    delay = doc('v039-wav-joint-phase', 'E041', 'wav_joint_phase.json')['timing_modes']['0']['0']['2']['polynomial_samples']
    word = np.asarray(doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')['cycle_folds'][0]['signed_template'])
    prior = doc('v111-wav-slot-case34-final', 'E113', 'wav_slot_case34.json')
    observations = doc('v112-wav-slot-frame-power-final', 'E114', 'wav_slot_frame_power.json')
    reference = doc('v110-wav-cycle-reacquisition-final', 'E112', 'wav_cycle_reacquisition.json')
    fixed_phase = prior['fixed_word_phase_subtracted']
    resources = np.asarray(geometry['reference_comparison']['resource_index_by_lane_and_slot_class'])
    carrier, _ = map(int, np.argwhere(resources == prior['target_resource_mod_60'])[0])
    slots = observations['observed_signs']['slot_order']
    all_symbols = build_symbols(word, resources, carrier, slots)
    relative = relative_symbols(all_symbols, fixed_phase, carrier, slots)
    with np.load(artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.npz'), allow_pickle=False) as data:
        harmonic_rows = data['harmonic_rows']
        mask = data['held_harmonic_mask']
    offset = geometry['fitted_offset_hz']
    basis = forward(harmonic_rows, offset).reshape(-1, len(CARRIERS), resources.shape[1])
    harmonics = harmonic_rows.ravel()
    frequency = harmonics * 750.
    centers = CARRIERS * GRID + offset
    scale = np.clip((frequency[:, None] - centers)/200_000., -2., 2.)
    target = abs(frequency - centers[carrier]) < GRID/2
    banks = []
    for degree in DEGREES:
        columns = design(basis, all_symbols.reshape(-1, *resources.shape), scale, degree)
        only_negative = columns[:len(word), :, :2*(degree + 1)]
        banks.append([(fit_bank(columns, fit), fit_bank(only_negative, fit & (harmonics < 0)))
                      for fit in (~mask, mask)])

    track = PROJECT_ROOT / 'experiments/E001/results/refined_track.npz'
    for path in (RAW_WAV, track):
        hashes[path] = sha256(path)
        if hashes[path] != grid['inputs_sha256'][path.relative_to(PROJECT_ROOT).as_posix()]:
            raise ValueError(f'Frozen source changed: {path.name}')
    with np.load(track, allow_pickle=False) as data:
        cw = data['quadratic_coefficients']
    fs, raw = wavfile.read(RAW_WAV, mmap=True)

    def window_spectra(start, alignment):
        samples, time, rate = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        return extract_frames(samples, time, fs, rate, alignment['offset_hz'], alignment['delay_samples'], harmonics)

    def evaluate(spectra, observed=None):
        all_fits = []
        for index, options in enumerate(banks):
            fits = []
            for arm, (bank, donor) in enumerate(options):
                projections = None if observed is None else np.column_stack(
                    [observed['targets'][str(slot)][index]['real_projection_by_arm'][1-arm] for slot in slots])
                fits.append(evaluate_arm(spectra, bank, donor, target, fixed_phase, len(word), relative, projections))
            all_fits.append(fits)
        return all_fits

    rows = []
    qualified = [row for row in prior['windows'] if row['arms'][0]['alignment']['qualified']]
    for index, (old, observed) in enumerate(zip(qualified, observations['windows'], strict=True)):
        alignment = old['arms'][0]['alignment']
        start = old['start_s']
        ordinal, spectra = window_spectra(start, alignment)
        if start != observed['window_start_s'] or ordinal.tolist() != observed['ordinals_in_window']:
            raise ValueError('E114 frame coordinates changed')
        fits = evaluate(spectra, observed)
        rows.append({'window_start_s': start, 'ordinals': ordinal.tolist(), 'time_group': index//GROUP_WINDOWS, 'fits': fits})
        if index % 8 == 0:
            print(start, [round(fits[0][0]['models'][name]['relative_held_error'][0], 5) for name in MODELS], flush=True)

    rng = np.random.default_rng(SEED)
    pre = reference['wav'][1]['arms'][0]
    gains = np.array([complex(*value) for value in pre['channel_gains']])
    unit_waveforms = design(basis, all_symbols.reshape(-1, *resources.shape), scale, 0) @ gains
    controls = []
    for start in POST:
        ordinal, background = window_spectra(start, alignment)
        phase = rng.choice(np.delete(np.arange(len(word)), fixed_phase), size=len(ordinal))
        states = rng.integers(2, size=len(ordinal))
        for label, chosen in (('word_shift', phase), ('three_slots', fixed_phase + len(word)*states)):
            common = np.exp(2j*np.pi*rng.random(len(ordinal)))
            for amplitude in (.5, 1.):
                signal = background + amplitude*common[:, None]*unit_waveforms[chosen]
                fits = evaluate(signal)
                controls.append({'window_start_s': start, 'injection': label, 'amplitude': amplitude,
                                 'phase_and_state_indices': chosen.tolist(), 'fits': fits})

    result = {'schema': 'starlink-wav-word-state/v1', 'seed': SEED, 'fixed_phase_subtracted': fixed_phase,
              'model_order': MODELS, 'region_order': REGIONS, 'carrier': int(CARRIERS[carrier]),
              'slot_classes': slots, 'region_counts_by_frequency_partition': [int((target & fit).sum()) for fit in (~mask, mask)],
              'reference_gain_window_s': reference['wav'][1]['start_s'],
              'input_sha256': {path.as_posix(): digest for path, digest in hashes.items()},
              'summary': summarize(rows, np.random.default_rng(SEED + 1)), 'windows': rows, 'synthetic_background': controls}
    destination = experiment_output('E116')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'wav_word_state.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(result['summary'], flush=True)


if __name__ == '__main__':
    main()
