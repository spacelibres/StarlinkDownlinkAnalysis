"""Replay the WAV word positions on frozen LEO receive-side apertures."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_blind_word import frame_signal
from audit_leo_pairs_live import REPORT
from audit_leo_tcode_iq_prediction import PERIOD, RESOURCES_PER_SLOT, WORD_SIZE, visible_carriers
from audit_narrowband_cycle import code_indices
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read


SIGN_FLOOR = .35


def word_indices(carriers, phase):
    return (code_indices(carriers)[:, None] + RESOURCES_PER_SLOT*np.arange(PERIOD) + phase) % WORD_SIZE


def target_coordinates(carriers, phase, positions):
    indices = word_indices(carriers, phase)
    hits = [np.argwhere(indices == position) for position in positions]
    if any(len(hit) != 1 for hit in hits):
        return None
    return np.vstack(hits)


def fit_state(data, carriers, phase, word, positions):
    coordinates = target_coordinates(carriers, phase, positions)
    if coordinates is None:
        return None
    carrier, slots = coordinates.T
    if not np.all(carrier == carrier[0]):
        raise ValueError('The three positions must lie on one carrier')
    k = carrier[0]
    symbols = word[word_indices(carriers, phase)].copy()
    target_atoms = data['features'][:, k, slots] * symbols[k, slots]
    symbols[k, slots] = 0
    anchor = np.einsum('nki,ki->nk', data['features'], symbols)
    # frame_signal has already removed the unmodulated sum of slots per carrier.
    active = np.any(symbols != symbols[:, :1], axis=1)
    anchor = anchor[:, active]
    pattern = target_atoms.sum(axis=1)
    design = np.column_stack((anchor, pattern))
    y, cal = data['target'], data['training']
    fitted, _, rank, _ = np.linalg.lstsq(design[cal], y[cal], rcond=1e-9)
    gain = fitted[np.count_nonzero(active[:k])]
    row = {'mapped_carrier': int(carriers[k]), 'slot_classes_of_slot_minus_2': slots.tolist(),
           'fit_rank': int(rank), 'fit_columns': design.shape[1]}
    if rank < design.shape[1] or abs(gain) <= 1e-9*np.linalg.norm(fitted):
        return {**row, 'qualified': False}
    delta = fitted[-1] / gain
    held = y[~cal] - anchor[~cal] @ fitted[:-1]
    trial = pattern[~cal] * gain
    power = float(np.vdot(trial, trial).real)
    if power <= 0.:
        return {**row, 'qualified': False}
    dot = np.vdot(trial, held)
    return {**row, 'qualified': True, 'donor_projection': float(delta.real),
            'donor_quadrature': float(delta.imag),
            'donor_sign': int(np.sign(delta.real)) if abs(delta.real) >= SIGN_FLOOR else 0,
            'held_projection': float(dot.real/power), 'held_quadrature': float(dot.imag/power),
            'held_dot': float(dot.real), 'held_pattern_power': power,
            'held_baseline_power': float(np.vdot(held, held).real)}


def transfer_scores(signs, dots, powers, baseline):
    signs, dots, powers, baseline = map(np.asarray, (signs, dots, powers, baseline))
    matched = signs != 0
    signs, dots, powers, baseline = (item[matched] for item in (signs, dots, powers, baseline))
    signs = signs.astype(int)
    row = {'used_frames': len(signs), 'positive_donor_frames': int(np.count_nonzero(signs > 0)),
           'negative_donor_frames': int(np.count_nonzero(signs < 0))}
    if not len(signs) or baseline.sum() <= 0.:
        return {**row, 'passes': False}

    def skill(candidate):
        return float((2*(candidate @ dots) - powers.sum())/baseline.sum())

    observed = skill(signs)
    positive = skill(np.ones(len(signs)))
    negative = skill(-np.ones(len(signs)))
    rotations = [skill(np.roll(signs, lag)) for lag in range(1, len(signs))]
    return {**row, 'observed_improvement': observed,
            'constant_positive_improvement': positive, 'constant_negative_improvement': negative,
            'nonzero_roll_improvements': rotations,
            'rolled_improvements_ge_observed': sum(value >= observed for value in rotations),
            'passes': min(row['positive_donor_frames'], row['negative_donor_frames']) >= 2
            and observed > max([0., positive, negative, *rotations])}


def evaluate_direction(direction, frames, receivers, phase_paths, word, positions, guards):
    donor_key, target_key = direction['donor'], direction['target']
    results = []
    for guard, path in zip(guards, phase_paths, strict=True):
        for offset in range(4):
            rows = []
            transfer = []
            same_coverage = same_carrier = 0
            for frame, phase in zip(frames, path['donor_selected_phase'], strict=True):
                row = {'frame': frame, 'frozen_donor_phase': phase}
                for key, label in ((donor_key, 'donor'), (target_key, 'target')):
                    carriers, data = receivers[key]
                    row[label] = fit_state(data[guard][frame], carriers, phase, word, (positions + offset) % WORD_SIZE)
                rows.append(row)
                donor, target = row['donor'], row['target']
                if donor is None or target is None:
                    continue
                same_coverage += 1
                same_carrier += donor['mapped_carrier'] == target['mapped_carrier']
                if donor['qualified'] and target['qualified']:
                    transfer.append((donor['donor_sign'], target['held_dot'], target['held_pattern_power'],
                                     target['held_baseline_power']))
            columns = zip(*transfer, strict=True) if transfer else ([], [], [], [])
            results.append({'guard_samples': guard, 'word_position_offset': offset,
                            'covered_frames': same_coverage, 'same_physical_carrier_frames': int(same_carrier),
                            'rows': rows, **transfer_scores(*columns)})
    return {'donor': donor_key, 'target': target_key, 'results': results}


def load_receiver(window, frames, guards, gates, old, frame_clock, templates):
    key = window['key']
    folder, array = key.split('/')
    selection = read(EXTERNAL_LEO_ROOT / REPORT / folder / 'selection.json')
    capture = next(row for row in selection['windows'] if f"iq_{row['number']}" == array)
    if (capture['probe']['edge'] != 'lower' or old['windows'][key]['edge'] != 'lower'
            or capture['observation_id'] != old['windows'][key]['observation_id']
            or selection['sample_rate_hz'] != window['rate_hz']):
        raise ValueError(f'Unexpected capture provenance for {key}')
    source = EXTERNAL_LEO_ROOT / REPORT / folder / 'selected_iq.npz'
    with np.load(source, allow_pickle=False) as files:
        iq = files[array]
    if hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest() != window['iq_sha256']:
        raise ValueError(f'Stale raw IQ {key}')
    carriers, hz = visible_carriers(window['rate_hz'], window['archived_cfo_hz'],
                                    templates.SUBCARRIER_SPACING_HZ, templates.qin_edge_pilot_indices('lower'))
    by_guard = {}
    for guard in guards:
        by_guard[guard] = {}
        for frame in frames:
            gate = next(g for g in window['frames'][frame] if g['guard_samples'] == guard)
            if not gate['positive']:
                raise ValueError(f'Frame {frame} no longer qualifies in {key}')
            by_guard[guard][frame] = frame_signal(iq, window, frame, gate, gates['calibration_blocks'],
                                                  frame_clock, templates, carriers, hz)
    return carriers, by_guard


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    hashes = {}

    def artifact(run, experiment, filename):
        return read(checked_artifact(run, experiment, filename, hashes))

    upstream = [artifact('v063-leo-tcode-pair-' + suffix, 'E065', 'leo_tcode_iq_prediction.json')
                for suffix in ('transfer', 'confirm')]
    if upstream[0] != upstream[1]:
        raise ValueError('Frozen E065 phase receipts disagree')
    e065 = upstream[0]
    e056 = artifact('v054-leo-cfo-confirm', 'E056', 'leo_cfo_aperture.json')
    e064 = artifact('v062-leo-pilot-phase-global', 'E064', 'leo_pilot_phase_transfer.json')
    wave = artifact('v112-wav-slot-frame-power-final', 'E114', 'wav_slot_frame_power.json')
    lattice = artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    anchor = artifact('v111-wav-slot-case34-final', 'E113', 'wav_slot_case34.json')
    tail = artifact('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    word = np.asarray(tail['cycle_folds'][0]['signed_template'])
    if word.tolist() != e065['sequence_controls'][0]:
        raise ValueError('LEO and WAV phase references differ')
    rows = np.asarray(lattice['reference_comparison']['resource_index_by_lane_and_slot_class'])
    lane = np.flatnonzero(np.any(rows == wave['target_resource_mod_60'], axis=1)).item()
    wav_carrier = lattice['carrier_indices'][lane]
    raw_positions = rows[lane, wave['observed_signs']['slot_order']]
    positions = (raw_positions - anchor['fixed_word_phase_subtracted']) % WORD_SIZE
    for relative, expected in e056['external_inputs_sha256'].items():
        path = EXTERNAL_LEO_ROOT / relative
        hashes[path] = sha256(path)
        if hashes[path] != expected:
            raise ValueError(f'External source changed: {relative}')
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT / 'src'), str(EXTERNAL_LEO_ROOT / 'tools')]
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    known_windows = {row['key']: row for row in e064['windows']}
    output = []
    for pair in e065['paired_receiver_phase_transfer']:
        frames = pair['frames']
        receiver_keys = [d['donor'] for d in pair['directions']]
        guards = [item['guard_samples'] for item in known_windows[receiver_keys[0]]['frames'][frames[0]]]
        receivers = {key: load_receiver(known_windows[key], frames, guards, e064, e056, FrameClock, templates)
                     for key in receiver_keys}
        directions = [evaluate_direction(d, frames, receivers, d['paths'][1::2], word, positions, guards)
                      for d in pair['directions']]
        output.append({'time_s': pair['time_s'], 'frames': frames, 'edge': 'lower',
                       'visible_fft_indices': {key: data[0].tolist() for key, data in receivers.items()},
                       'covers_literal_wav_carrier': {key: wav_carrier in data[0]-1024 for key, data in receivers.items()},
                       'directions': directions})
        for direction in directions:
            print(pair['time_s'], direction['donor'], [
                (row['guard_samples'], row['word_position_offset'], row['covered_frames'], row['used_frames'],
                 row.get('observed_improvement'), row['passes']) for row in direction['results']], flush=True)

    result = {'schema': 'starlink-leo-common-state/v1', 'wav_signed_carrier': wav_carrier,
              'wav_resource_positions': raw_positions.tolist(), 'template_positions': positions.tolist(),
              'input_sha256': {path.as_posix(): digest for path, digest in hashes.items()},
              'training_blocks': e064['calibration_blocks'], 'held_blocks': e064['held_blocks'],
              'sign_floor': SIGN_FLOOR, 'pairs': output}
    destination = experiment_output('E115')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'leo_common_state.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
