"""Constrain the narrowband tail onset and audit wideband phase ambiguity."""

from collections import Counter, defaultdict
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
from audit_wav_slot_frame_power import extract_frames
from audit_wav_tessellation import CARRIERS, CP, EPOCH, GRID, RATE, SLOT, forward
from audit_wav_word_state import fit_bank, score_selected
from layout_core import NONPILOT
from receiver_models import pn_reference


PERIOD = 60
PHASES = (0, 8, 16)
SEED = 20261121120
TOTAL = 300*len(NONPILOT)
FOLDS = 4
MIN_FRAMES = 6
LAST_EARLY_SLOT = 51


def first_supported_slots(endpoints, ordinals):
    delta = np.asarray(endpoints)[:, None] - np.asarray(ordinals)
    return 2 + np.clip((delta + len(NONPILOT)-1)//len(NONPILOT), 0, 300)


def endpoint_support(phase, ordinals):
    end = np.arange(phase, TOTAL, PERIOD)
    support = first_supported_slots(end, ordinals)
    unique, inverse = np.unique(support, axis=0, return_inverse=True)
    return unique, [end[inverse == k] for k in range(len(unique))]


def accepted_phase(row):
    signal = row['signal']
    if signal['kind'] == 'periodic':
        return signal['periodic_phase']
    short = row.get('secondary_short', {})
    return short['phase'] if short.get('accepted') else None


def grouped_clock_rows(tails, clock):
    labels = {row['frame']: row for row in clock}
    groups = {}
    prior = {}
    rows = []

    def root(group):
        while groups[group] != group:
            group = groups[group]
        return group

    for row in tails:
        frame = row['frame']
        phase = accepted_phase(row)
        if phase is None:
            continue
        label = labels[frame]
        # Four contiguous 13-frame source pages, with repeated time labels linked below.
        source_group = row['group']//52
        groups.setdefault(source_group, source_group)
        clock_key = label['base_gps_s'], label['counter']
        if clock_key in prior:
            groups[root(source_group)] = root(prior[clock_key])
        prior[clock_key] = source_group
        end = row['comparison']['separate_rounding_end']
        rows.append({'frame': frame, 'phase': phase, 'end': end, 'source_group': source_group,
                     'descriptor_sequence': [(i['mcs_candidate'], i['quantity']) for i in row['comparison']['descriptors']],
                     'counter': label['counter'], 'anchor': label['base_gps_s'],
                     'direct_clock_bits': label['direct_clock_bits']})
    for row in rows:
        row['linked_group'] = root(row['source_group'])
    return rows


def clock_predictions(rows, rng):
    group = np.array([r['linked_group'] for r in rows])
    frames = np.array([r['frame'] for r in rows])
    phase = np.array([r['phase'] for r in rows])
    counter = np.array([r['counter'] for r in rows])
    assignments = np.array_split(rng.permutation(np.unique(group)), FOLDS)
    result = []
    for held_groups in assignments:
        held = np.isin(group, held_groups)
        baseline = int(np.bincount(phase[~held], minlength=60).argmax())
        predicted = {'constant': np.full(int(held.sum()), baseline)}
        for mod in (4, 15, 60, 750):
            train_keys = counter[~held] % mod
            pairs = train_keys*PERIOD + phase[~held]
            by_mod = np.bincount(pairs, minlength=mod*PERIOD).reshape(mod, PERIOD)
            pred = np.where(by_mod.sum(axis=1) > 0, by_mod.argmax(axis=1), baseline)
            predicted[f'counter_mod_{mod}'] = pred[counter[held] % mod]
        a, b = np.indices((PERIOD, PERIOD)).reshape(2, -1)
        fit = (a[:, None]*counter[~held] + b[:, None]) % PERIOD
        ai = int(np.sum(fit == phase[None, ~held], axis=1).argmax())
        predicted['affine_mod_60'] = (a[ai]*counter[held] + b[ai]) % PERIOD
        truth = phase[held]
        direct = np.array([r['direct_clock_bits'] == 72 for r in rows])[held]
        result.append({'held_groups': held_groups.tolist(), 'held_frames': frames[held].tolist(),
                       'affine_multiplier_and_offset': [int(a[ai]), int(b[ai])],
                       'n': len(truth), 'n_direct_clock_bits': int(direct.sum()),
                       'correct': {k: int(np.count_nonzero(v == truth)) for k, v in predicted.items()},
                       'correct_direct_clock_bits': {k: int(np.count_nonzero((v == truth) & direct))
                                                     for k, v in predicted.items()}})
    return result


def audit_archive(wide, clock, rng):
    rows = grouped_clock_rows(wide['qin'], clock['archive'])
    supported = Counter(r['phase'] for r in rows)
    collisions = {}
    for phase in PHASES:
        candidates = [r for r in rows if r['phase'] == phase]
        endpoints = sorted({r['end'] for r in candidates})
        paths = {tuple(r['descriptor_sequence']) for r in candidates}
        collisions[phase] = {'frames': len(candidates), 'distinct_endpoints': endpoints,
                             'distinct_descriptor_sequences': len(paths),
                             'boundary_slots': [2+e//len(NONPILOT) for e in endpoints]}
    by_clock = defaultdict(list)
    for row in rows:
        by_clock[row['anchor'], row['counter']].append(row['phase'])
    return {'phase_count': dict(sorted(supported.items())), 'dominant_phase_collisions': collisions,
            'equal_endpoint_mod_60': sum(r['phase'] == r['end'] % PERIOD for r in rows),
            'complete_rows': len(rows), 'records': rows, 'clock_prediction_folds': clock_predictions(rows, rng),
            'duplicated_clock_labels': sum(len(v) > 1 for v in by_clock.values()),
            'duplicated_label_phase_conflicts': sum(len(set(v)) > 1 for v in by_clock.values()),
            'legacy_terminal_shape_counts': dict(Counter(r['matched_tones']['kind'] or 'erased' for r in wide['legacy']))}


def tail_columns(harmonics, offset, carriers, symbols, start_slots):
    frequencies = harmonics * 750.
    slots = np.arange(2, 302)
    time = EPOCH/RATE + SLOT*slots
    window = np.exp(-2j*np.pi*frequencies[:, None]*time)
    out = []
    for carrier, code, start in zip(carriers, symbols, start_slots.T, strict=True):
        phase = pn_reference(int(carrier), slots)*np.exp(2j*np.pi*(offset*SLOT*slots - GRID*carrier*CP))
        q = (GRID*carrier+offset-frequencies)*SLOT
        contribution = window*phase*code[slots % 15]*(np.sinc(q)*np.exp(1j*np.pi*q))[:, None]
        suffix = np.column_stack((np.cumsum(contribution[:, ::-1], axis=1)[:, ::-1],
                                  np.zeros(len(frequencies))))
        out.append(suffix[:, start-2].T)
    return np.stack(out, axis=-1)


def cohere(spectra, donor_columns, donor):
    train = donor_columns[donor]
    values = spectra[:, donor]
    initial = np.linalg.lstsq(train, values.mean(axis=0), rcond=1e-8)[0]
    observed = values @ (train @ initial).conj()
    normalized = spectra * (observed.conj()/np.maximum(abs(observed), 1e-30))[:, None]
    return normalized.mean(axis=0)


def select_by_support(scores, early):
    early_ids = np.flatnonzero(early)
    late_ids = np.flatnonzero(~early)
    return {'selected': int(scores.argmax()), 'best_early': int(early_ids[scores[early].argmax()]),
            'best_late': int(late_ids[scores[~early].argmax()])}


def evaluate_model(bank, spectra, split, slot9, early):
    objectives = bank.objectives(spectra)
    cal = abs(spectra[:, bank.calibration])**2
    cal_power = np.maximum(cal.sum(axis=1), 1e-30)
    region = ~bank.calibration
    results = []
    for train in (split, ~split):
        if not train.any() or train.all():
            raise ValueError('No independent calibration and held window groups')
        strength = (objectives[:, train]/cal_power[train]).mean(axis=1)
        chosen = {**select_by_support(strength, early), 'start_at_9': slot9}
        held = spectra[~train]
        held_power = float(np.sum(abs(held[:, region])**2))
        scores = {name: float(score_selected(bank, held, np.full(len(held), idx), region).sum()/held_power)
                  for name, idx in chosen.items()}
        results.append({'chosen': chosen['selected'], 'selected_support_indices': chosen, 'held_relative_error': scores,
                        'calibration_window_indices': np.flatnonzero(train).tolist(),
                        'held_window_indices': np.flatnonzero(~train).tolist(),
                        'calibration_peak_power_fraction': float(strength.max())})
    return results


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, filename):
        return checked_artifact(run, exp, filename, hashes)

    def doc(run, exp, filename):
        return read(artifact(run, exp, filename))

    archive = audit_archive(doc('v065-full-tail-phase-final', 'E067', 'tail_phase_audit.json'),
                            doc('v095-code5-clock-final', 'E097', 'code5_clock_correction.json'),
                            np.random.default_rng(SEED))
    state = doc('v114-wav-word-state', 'E116', 'wav_word_state.json')
    if state != doc('v114-wav-word-state-confirm', 'E116', 'wav_word_state.json'):
        raise ValueError('Frozen WAV word-state inputs differ')
    geo = doc('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    grid = doc('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json')
    delay = doc('v039-wav-joint-phase', 'E041', 'wav_joint_phase.json')['timing_modes']['0']['0']['2']['polynomial_samples']
    alignment = doc('v111-wav-slot-case34-final', 'E113', 'wav_slot_case34.json')
    symbols = np.asarray(doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')['cycle_folds'][0]['signed_template'])
    gains = doc('v110-wav-cycle-reacquisition-final', 'E112', 'wav_cycle_reacquisition.json')['wav'][1]['arms'][0]['channel_gains']
    with np.load(artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.npz'), allow_pickle=False) as data:
        harmonic_rows = data['harmonic_rows']
        reserved = data['held_harmonic_mask']
    harmonics = harmonic_rows.ravel()
    pos, neg = harmonics > 0, harmonics < 0
    offset = geo['fitted_offset_hz']
    lattice = forward(harmonic_rows, offset).reshape(len(harmonics), len(CARRIERS), 15)
    resources = np.asarray(geo['reference_comparison']['resource_index_by_lane_and_slot_class'])
    c2 = CARRIERS > 0
    ordinals = np.array([np.flatnonzero(NONPILOT == k).item() for k in CARRIERS[c2]])
    word_by_phase = {p: symbols[(resources-p) % PERIOD] for p in PHASES}
    donor_banks = {p: np.einsum('nks,ks->nk', lattice[:, ~c2], w[~c2]) for p, w in word_by_phase.items()}
    track = PROJECT_ROOT / 'experiments/E001/results/refined_track.npz'
    for path in (RAW_WAV, track):
        hashes[path] = sha256(path)
        if hashes[path] != grid['inputs_sha256'][path.relative_to(PROJECT_ROOT).as_posix()]:
            raise ValueError(f'Frozen source changed: {path.name}')
    with np.load(track, allow_pickle=False) as data:
        cw = data['quadratic_coefficients']
    fs, raw = wavfile.read(RAW_WAV, mmap=True)

    def spectra_at(start, receiver):
        y, t, rate = sample_window(raw, fs, start, cw, grid['frequency_grid_hz'], delay)
        return extract_frames(y, t, fs, rate, receiver['offset_hz'], receiver['delay_samples'], harmonics)

    selected = {p: [] for p in PHASES}
    alignments = {w['start_s']: w['arms'][0]['alignment'] for w in alignment['windows']}
    for row in state['windows']:
        start = row['window_start_s']
        ordinals_in_window, spectra = spectra_at(start, alignments[start])
        if ordinals_in_window.tolist() != row['ordinals']:
            raise ValueError('Frame coordinates differ from E116')
        choices = np.array([arm['models']['negative_side_shift']['phase'] for degree in row['fits'] for arm in degree])
        for p in PHASES:
            take = np.flatnonzero((choices == p).all(axis=0))
            if len(take) >= MIN_FRAMES:
                y = cohere(spectra[take], donor_banks[p], neg)[pos]
                selected[p].append((row['time_group'], start, ordinals_in_window[take].tolist(), y))

    rng = np.random.default_rng(SEED+1)
    time_groups = np.unique([r['time_group'] for r in state['windows']])
    learning = rng.permutation(time_groups)[:len(time_groups)//2]
    receiver = alignments[state['windows'][-1]['window_start_s']]
    backgrounds = [spectra_at(start, receiver)[1][:24, pos].mean(axis=0) for start in (13.68, 14.48, 15.48)]
    injected_gains = np.array([complex(*value) for value in gains])[c2]
    results = []
    for p in PHASES:
        begins, groups = endpoint_support(p, ordinals)
        if not np.any(np.all(begins == 9, axis=1)):
            raise ValueError('The fixed slot-9 baseline is not representable')
        baseline_index = np.flatnonzero(np.all(begins == 9, axis=1)).item()
        columns = tail_columns(harmonics[pos], offset, CARRIERS[c2], word_by_phase[p][c2], begins)
        banks = [fit_bank(columns, fit) for fit in (~reserved[pos], reserved[pos])]
        early = (begins <= LAST_EARLY_SLOT).all(axis=1)
        group, starts, frames, data = zip(*selected[p], strict=True)
        measurements = [evaluate_model(bank, np.asarray(data), np.isin(group, learning), baseline_index, early)
                        for bank in banks]
        control = []
        for bg, known_slot in zip(backgrounds, (26, 124, 239), strict=True):
            known = np.flatnonzero(np.all(begins == known_slot, axis=1)).item()
            for amplitude in (.5, 1.):
                model = bg + amplitude*(columns[known] @ injected_gains)
                trials = []
                for bank in banks:
                    fit = bank.calibration
                    chosen = select_by_support(bank.objectives(model[None])[:, 0], early)
                    trials.append({'chosen': chosen['selected'], 'selected_support_indices': chosen,
                                   'held_errors': {name: score_selected(bank, model[None], [idx], ~fit).item()
                                                   for name, idx in {**chosen, 'known': known}.items()}})
                control.append({'known_index': known, 'amplitude': amplitude, 'arms': trials})
        groups_compact = [{'start_slot_by_carrier': row.tolist(), 'min_endpoint': int(ends[0]),
                           'max_endpoint': int(ends[-1]), 'equivalent_endpoint_count': len(ends)}
                          for row, ends in zip(begins, groups, strict=True)]
        print('phase', p, 'windows', len(starts), 'frames', sum(map(len, frames)),
              'onsets', [begins[r['chosen']].tolist() for a in measurements for r in a],
              'controls', [(begins[c['known_index']].tolist(), [begins[r['chosen']].tolist() for r in c['arms']])
                           for c in control], flush=True)
        results.append({'phase': p, 'window_times_s': starts, 'frame_ordinals_in_window': frames,
                        'early_equivalent_endpoints': sum(row['equivalent_endpoint_count']
                                                          for row, flag in zip(groups_compact, early, strict=True) if flag),
                        'support_bank': groups_compact, 'calibration_groups': list(map(int, learning)),
                        'four_held_branches': measurements, 'synthetic_controls': control})
    report = {'schema': 'starlink-wav-tail-support/v2', 'seed': SEED, 'input_sha256': {str(p): h for p, h in hashes.items()},
              'last_early_slot': LAST_EARLY_SLOT,
              'archive': archive, 'positive_carriers': CARRIERS[c2].tolist(), 'wav_tail_support': results}
    destination = experiment_output('E117')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'wav_tail_support.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
