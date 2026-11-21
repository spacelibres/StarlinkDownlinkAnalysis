"""Measure held-sample transfer of individual pilot phases in archived LEO IQ."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_pairs_live import REPORT
from audit_leo_paired_slots import FRAMES, GUARDS, SEED, partition_groups
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read


CFO_GRID_HZ = np.arange(-1000., 1001., 25.)
CONTROL_ROLL = 17
FIRST_SLOT = 32
BLOCK_SLOTS = 45


def profile_pilots(basis, samples, time, training):
    cal = basis[training]
    y = samples[training]
    angles = np.exp(-2j*np.pi*time[training, None]*CFO_GRID_HZ)
    gram = cal.conj().T @ cal
    rhs = cal.conj().T @ (y[:, None]*angles)
    gains = np.linalg.solve(gram, rhs)
    energies = np.sum((rhs.conj()*gains).real, axis=0)
    winner = int(np.argmax(energies))
    frequency = CFO_GRID_HZ[winner]
    learned = gains[:, winner]

    held = basis[~training]
    target = samples[~training]*np.exp(-2j*np.pi*time[~training]*frequency)
    gram_held = held.conj().T @ held
    measured = np.linalg.solve(gram_held, held.conj().T @ target)
    # Squared norm of each pilot after projection away from the other seven.
    weights = 1/np.linalg.inv(gram_held).diagonal().real
    residual = target-held @ learned
    held_energy = np.vdot(target, target).real
    return {
        'cfo_hz': float(frequency), 'cfo_at_grid_boundary': winner in (0, len(CFO_GRID_HZ)-1),
        'calibration_gain': learned, 'held_gain': measured, 'held_target_weight': weights,
        'held_total_energy': float(held_energy),
        'held_pilot_energy_improvement': float(1-np.vdot(residual, residual).real/held_energy),
    }


def compare_phase(calibration, held, weights):
    # Equal spacing: interpolate at the interior and extrapolate at the ends.
    left = np.r_[1, np.arange(6), 5]
    right = np.r_[2, np.arange(2, 8), 6]
    fraction = (np.arange(8)-left)/(right-left)
    left_angle = np.angle(calibration[left])
    neighbor_slope = np.angle(calibration[right]*calibration[left].conj())
    smooth = left_angle+fraction*neighbor_slope
    delta = np.angle(calibration*np.exp(-1j*smooth))
    held_delta = np.angle(held*np.exp(-1j*smooth))
    magnitude = abs(calibration)
    tone = np.arange(8)
    line = np.column_stack((np.ones(8), tone-tone.mean()))
    global_smooth = np.empty(8)
    for k in tone:
        use = tone != k
        amplitude = magnitude[use]
        phase = np.unwrap(np.angle(calibration[use]))
        coefficients = np.linalg.lstsq(line[use]*amplitude[:, None], phase*amplitude, rcond=None)[0]
        global_smooth[k] = line[k] @ coefficients
    global_delta = np.angle(calibration*np.exp(-1j*global_smooth))

    def error(phase):
        return weights*abs(held-magnitude*np.exp(1j*phase))**2

    return {
        'phase_delta_rad': delta.tolist(), 'held_delta_rad': held_delta.tolist(),
        'neighbor_phase_step_rad_per_tone': (neighbor_slope/(right-left)).tolist(),
        'calibration_amplitude': magnitude.tolist(), 'held_amplitude': abs(held).tolist(),
        'zero_error': (weights*abs(held)**2).tolist(), 'smooth_error': error(smooth).tolist(),
        'individual_error': error(smooth+delta).tolist(), 'reversed_error': error(smooth-delta).tolist(),
        'global_phase_delta_rad': global_delta.tolist(),
        'global_held_delta_rad': np.angle(held*np.exp(-1j*global_smooth)).tolist(),
        'global_smooth_error': error(global_smooth).tolist(),
    }


def geometry(clock, origin, rate, slot_s, cp_s):
    first = np.ceil(origin+(2*slot_s+cp_s)*rate+.6)
    time = clock.time-clock.time[0]+(first-origin)/rate
    slot = np.floor(time/slot_s+1e-10).astype(int)
    local = time-slot*slot_s
    tolerance = 1e-12
    if (slot.min() != 2 or slot.max() != 301 or
            local.min() < cp_s+.6/rate-tolerance or local.max() > slot_s-.6/rate+tolerance):
        raise ValueError('Original FrameClock sample support changed')
    return slot, local


def frame_scores(iq, rate, origin, cfo, edge, train, frame_clock, templates):
    actual = frame_clock(iq, rate, origin, cfo, edge)
    control = frame_clock(iq, rate, origin, cfo, edge, roll=CONTROL_ROLL)
    slot_s = templates.OFDM_SYMBOL_DURATION_S
    cp_s = templates.CYCLIC_PREFIX_DURATION_S
    slot, local = geometry(actual, origin, rate, slot_s, cp_s)
    blocks = (slot-FIRST_SLOT)//BLOCK_SLOTS
    results = []
    for guard in GUARDS:
        keep = ((blocks >= 0) & (local >= cp_s+guard/rate-1e-12) &
                (local <= slot_s-guard/rate+1e-12))
        calibration = np.isin(blocks[keep], train)
        fits = [
            profile_pilots(model.basis[keep], model.values[keep], model.time[keep], calibration)
            for model in (actual, control)
        ]
        true, shifted = fits
        true_skill = true['held_pilot_energy_improvement']
        wrong_skill = shifted['held_pilot_energy_improvement']
        result = compare_phase(true['calibration_gain'], true['held_gain'], true['held_target_weight'])
        result.update({
            'guard_samples': guard, 'calibration_samples': int(calibration.sum()),
            'held_samples': int((~calibration).sum()), 'actual_cfo_hz': true['cfo_hz'],
            'wrong_cfo_hz': shifted['cfo_hz'], 'cfo_at_grid_boundary': true['cfo_at_grid_boundary'],
            'held_total_energy': true['held_total_energy'], 'actual_waveform_improvement': true_skill,
            'wrong_waveform_improvement': wrong_skill, 'positive': true_skill > max(0., wrong_skill),
        })
        results.append(result)
    return results


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    inputs = {}
    reference = checked_artifact('v054-leo-cfo-confirm', 'E056', 'leo_cfo_aperture.json', inputs)
    prior = read(reference)
    for rel, expected in prior['external_inputs_sha256'].items():
        path = EXTERNAL_LEO_ROOT/rel
        inputs[path] = sha256(path)
        if inputs[path] != expected:
            raise ValueError(f'Archived E056 source mismatch: {rel}')
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT/'src'), str(EXTERNAL_LEO_ROOT/'tools')]
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    train, held = partition_groups()
    seen = {}
    aliases = {}
    for folder in ('DSX01-S05', 'paired_RX'):
        source = EXTERNAL_LEO_ROOT/REPORT/folder
        selection = read(source/'selection.json')
        rate = selection['sample_rate_hz']
        with np.load(source/'selected_iq.npz', allow_pickle=False) as archive:
            for entry in selection['windows']:
                number = entry['number']
                key = f'{folder}/iq_{number}'
                iq = archive[f'iq_{number}']
                digest = hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest()
                if digest in seen:
                    aliases[key] = seen[digest]['key']
                    continue
                winner = entry['winner']
                cfo = winner['fractional_tracking_cfo_hz']
                epoch = winner['integer_epoch_sample']+winner['fractional_epoch_offset_samples']
                edge = entry['probe']['edge']
                lookup = prior['windows'][key]
                if (digest, epoch, cfo, edge, rate) != (
                        lookup['iq_sha256_complex64_le'], lookup['epoch_sample'], lookup['original_cfo_hz'],
                        lookup['edge'], lookup['sample_rate_hz']):
                    raise ValueError(f'E056 coordinates changed: {key}')
                frames = [
                    frame_scores(iq, rate, epoch+f*rate/templates.FRAME_RATE_HZ, cfo, edge,
                                 train, FrameClock, templates)
                    for f in range(FRAMES)
                ]
                seen[digest] = {
                    'key': key, 'iq_sha256': digest, 'rate_hz': rate, 'epoch_sample': epoch,
                    'archived_cfo_hz': cfo, 'receiver': entry['receiver'], 'context_time_s': entry['context_time_s'],
                    'carrier_indices': list(templates.qin_edge_pilot_indices(edge)), 'frames': frames,
                }
                print(key, 'joint positive frames', sum(all(r['positive'] for r in f) for f in frames),
                      'of', FRAMES, flush=True)
    if aliases != prior['duplicate_iq']:
        raise ValueError('E056 duplicate window assignments changed')
    out = experiment_output('E064')
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'leo-pilot-phase-transfer/v2',
        'inputs_sha256': {p.as_posix(): d for p, d in inputs.items()},
        'seed': SEED, 'calibration_blocks': train.tolist(), 'held_blocks': held.tolist(),
        'first_slot': FIRST_SLOT, 'block_slots': BLOCK_SLOTS, 'wrong_sequence_roll': CONTROL_ROLL,
        'cfo_grid_hz': CFO_GRID_HZ.tolist(), 'duplicate_windows': aliases, 'windows': list(seen.values()),
    }
    (out/'leo_pilot_phase_transfer.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
