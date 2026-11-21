"""Use the primary-grid cyclic prefix to test pilot preservation on a second grid."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples
from audit_ut_repeated_data import checked_artifact, read
from audit_ut_weak_frames import CP, FFT_SIZE, FRAME, RATE, SLOT
from code5_core import PILOTS, sinc_sample
from receiver_models import pn_reference


SEED = 2026112159
SLOTS = np.arange(67, 211)
CP_TAPS = np.arange(10, CP-10, 2)
LAGS = (0, -47, 47)
TABLE = np.column_stack([pn_reference(int(k), np.arange(302)) for k in PILOTS])
PER_BLOCK = 12
FFT_TRAIN_OFFSETS = (3, 6, 9)


def pilot_features(positions, epoch, scale, relative_hz, *, roll=0):
    time = (positions-epoch)/scale
    slot = np.floor(time/SLOT).astype(int)
    if (slot < 2).any() or (slot > 301).any():
        raise ValueError('Selected time is outside the candidate OFDM body')
    local = time-slot*SLOT-CP
    phase = local[..., None]*PILOTS/FFT_SIZE + relative_hz*(positions-epoch)[..., None]/RATE
    return TABLE[slot+roll]*np.exp(2j*np.pi*phase)


def train_gain(values, basis, train):
    fit = basis[train].reshape(-1, basis.shape[-1])
    gain, _, rank, singular = np.linalg.lstsq(fit, values[train].ravel(), rcond=None)
    if rank != basis.shape[-1]:
        raise ValueError('Pilot template is rank deficient')
    return gain, float(singular[0]/singular[-1])


def prediction_score(observed, modeled):
    e = float(np.vdot(observed, observed).real)
    p = float(np.vdot(modeled, modeled).real)
    cross = float(np.vdot(modeled, observed).real)
    return {'improvement': (2*cross-p)/max(e, 1e-30),
            'coherence': cross/max(np.sqrt(e*p), 1e-30)}


def predict(values, basis, train):
    gains, condition = train_gain(values, basis, train)
    held_template = basis[~train] @ gains
    return {**prediction_score(values[~train], held_template), 'condition': condition}


def raw_samples(iq, positions, carrier):
    return sinc_sample(iq, positions.ravel()).reshape(positions.shape)*np.exp(
        -2j*np.pi*carrier*positions/RATE
    )


def prefix_positions(epoch, scale):
    return epoch + scale*(SLOTS[:, None]*SLOT + CP_TAPS)


def calibration_fft(iq, epoch, scale, hz, a_epoch, a_scale, a_hz):
    """Take complete useful symbols strictly inside each physical time block."""
    block_starts = SLOTS[::PER_BLOCK]
    approximate = a_epoch+a_scale*SLOT*(block_starts[:, None]+FFT_TRAIN_OFFSETS)
    slots = np.floor(((approximate+a_scale*SLOT/2)-epoch)/scale/SLOT).astype(int)
    first = epoch+(slots*SLOT+CP)*scale
    last = first+(FFT_SIZE-1)*scale
    block_first = a_epoch+block_starts*a_scale*SLOT
    block_last = block_first+PER_BLOCK*a_scale*SLOT
    if np.any(first < block_first[:, None]+8) or np.any(last > block_last[:, None]-9):
        raise ValueError('A calibration FFT crosses a whole-block boundary')
    if slots.min() < 2 or slots.max() > 300:
        raise ValueError('Calibration slots outside the OFDM body')
    raw = np.array([raw_samples(iq, p+scale*np.arange(FFT_SIZE), hz) for p in first.ravel()])
    values = np.fft.fft(raw, axis=1)[:, PILOTS % FFT_SIZE]/FFT_SIZE
    values = values.reshape(*slots.shape, len(PILOTS))
    return values*np.exp(2j*np.pi*(hz-a_hz)*epoch/RATE), slots


def fft_gain(coefficients, slots, training, *, roll=0):
    train_blocks = training[::PER_BLOCK]
    return (coefficients[train_blocks]*TABLE[slots[train_blocks]+roll].conj()).mean(axis=(0, 1))


def evaluate(iq, a, a_scale, b, b_scale, training):
    a_hz = a['cfo_hz']
    a_time = a['epoch_480_samples']
    b_time = b['epoch_480_samples']
    b_rel_hz = b['cfo_hz']-a_hz
    early_times = prefix_positions(a_time, a_scale)
    early = raw_samples(iq, early_times, a_hz)
    a_design = pilot_features(early_times, a_time, a_scale, 0.)
    basis_args = (b_time, b_scale, b_rel_hz)
    b_head = pilot_features(early_times, *basis_args)
    b_head_roll = pilot_features(early_times, *basis_args, roll=1)
    a_fft, a_fft_slots = calibration_fft(iq, a_time, a_scale, a_hz, a_time, a_scale, a_hz)
    b_fft, b_fft_slots = calibration_fft(iq, b_time, b_scale, b['cfo_hz'], a_time, a_scale, a_hz)

    pairs = []
    for delta in LAGS:
        late_times = early_times + a_scale*(FFT_SIZE+delta)
        late = raw_samples(iq, late_times, a_hz)
        difference = late-early
        b_delta = pilot_features(late_times, *basis_args) - b_head
        b_delta_roll = pilot_features(late_times, *basis_args, roll=1) - b_head_roll
        results = []
        for train in training:
            test = ~train
            a_gains, _ = train_gain(early, a_design, train)
            a_template = a_design[test] @ a_gains
            a_before = abs(np.vdot(a_template, early[test]))
            a_after = abs(np.vdot(a_template, difference[test]))
            a_fft_prediction = a_design[test] @ fft_gain(a_fft, a_fft_slots, train)
            b_gain = fft_gain(b_fft, b_fft_slots, train)
            wrong_b_gain = fft_gain(b_fft, b_fft_slots, train, roll=1)
            fft_before = abs(np.vdot(a_fft_prediction, early[test]))
            fft_after = abs(np.vdot(a_fft_prediction, difference[test]))
            total_energy = float(np.vdot(early[test], early[test]).real + np.vdot(late[test], late[test]).real)
            spared_energy = 1-float(np.vdot(difference[test], difference[test]).real)/total_energy
            results.append({
                'copy_energy_difference_fraction': spared_energy,
                'a_pilot_after_to_before_amplitude': float(a_after/max(a_before, 1e-30)),
                'a_pilot_prefix': prediction_score(early[test], a_template),
                'b_pilot_prefix': predict(early, b_head, train),
                'b_wrong_pilot_prefix': predict(early, b_head_roll, train),
                'b_pilot_difference': predict(difference, b_delta, train),
                'b_wrong_pilot_difference': predict(difference, b_delta_roll, train),
                'fft_calibration': {
                    'a_pilot_after_to_before_amplitude': float(fft_after/max(fft_before, 1e-30)),
                    'a_pilot_prefix': prediction_score(early[test], a_fft_prediction),
                    'b_pilot_prefix': prediction_score(early[test], b_head[test] @ b_gain),
                    'b_wrong_pilot_prefix': prediction_score(early[test], b_head_roll[test] @ wrong_b_gain),
                    'b_pilot_difference': prediction_score(difference[test], b_delta[test] @ b_gain),
                    'b_wrong_pilot_difference': prediction_score(difference[test], b_delta_roll[test] @ wrong_b_gain),
                },
            })
        pairs.append({'lag_error_480_samples': delta, 'validation': results})
    return {'a_b_delay_480_samples': b_time-a_time, 'b_relative_carrier_hz': b_rel_hz, 'pairs': pairs}


def main():
    require_run_context()
    inputs = {}
    ut = read(checked_artifact('v022-ut-replay', 'E014', 'ut_transfer.json', inputs))
    dual = read(checked_artifact('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json', inputs))
    fit = read(checked_artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json', inputs))
    path = ROOT/SOURCE
    iq = load_samples(path, ut)
    inputs[path] = ut['source_sha256']
    b_timing = fit['folds'][0]['timing']
    a_scale = dual['sampling_scale']
    a_by_tick = {r['tick']: r for r in dual['records']}
    b_by_tick = {r['tick']: r['sync'] for r in dual['offset_grid']['records']
                 if r.get('sync', {}).get('sync_pair_passed')}
    overlap = sorted(set(b_by_tick) & {tick for tick, r in a_by_tick.items() if r['sync']['sync_pair_passed']})
    if overlap != fit['overlap_ticks']:
        raise ValueError('Archive overlap assignments changed')
    origin = ut['candidates'][0]['pss']['epoch']
    a_only = {round((r['pss']['epoch']-origin)/FRAME): r for r in ut['candidates']
              if r.get('h2', {}).get('crc_ok')}
    if set(a_only) & set(b_by_tick):
        raise ValueError('Expected the strong primary references to be outside offset detections')
    blocks = np.arange(len(SLOTS))//PER_BLOCK
    assigned = np.sort(np.random.default_rng(SEED).permutation(len(np.unique(blocks)))[:6])
    train = np.isin(blocks, assigned)
    rows = []
    for label, ticks in (('overlap', overlap), ('offset_detection_only', sorted(set(b_by_tick)-set(overlap))),
                         ('primary_strong_only_detected', sorted(a_only))):
        for tick in ticks:
            main = a_by_tick[tick]
            if label == 'offset_detection_only':
                a = {'epoch_480_samples': main['predicted_epoch_480_samples'],
                     'cfo_hz': main['predicted_cfo_hz']}
            elif label == 'primary_strong_only_detected':
                r = a_only[tick]
                a = {'epoch_480_samples': r['pss']['epoch'], 'cfo_hz': r['cfo_hz']}
            else:
                a = main['sync']
            if tick in b_by_tick:
                b = {**b_by_tick[tick], 'cfo_hz': b_by_tick[tick]['cfo_hz']+b_timing['cfo_residual_hz']}
            else:
                b = {'epoch_480_samples': tick*FRAME+float(np.polyval(dual['offset_grid']['epoch_residual_fit'], tick)),
                     'cfo_hz': main['predicted_cfo_hz']+b_timing['cfo_residual_hz']}
            row = {'tick': tick, 'detection_group': label, 'a': a, 'b': b}
            times = prefix_positions(a['epoch_480_samples'], a_scale)
            guard_ok = times[0, 0] >= 8
            guard_ok &= times[-1, -1] + a_scale*(FFT_SIZE+max(LAGS))+9 < len(iq)
            if guard_ok:
                row.update(status='evaluated', **evaluate(
                    iq, a, a_scale, b, b_timing['sampling_scale'], (train, ~train)
                ))
            else:
                row.update(status='incomplete_fixed_slot_support')
            rows.append(row)
            print(label, tick, row['status'], flush=True)
    out = experiment_output('E059')
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'starlink-overlap-cp-cancellation/v2',
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        'seed': SEED, 'calibration_blocks_first_split': assigned.tolist(),
        'primary_slots': SLOTS.tolist(), 'cp_taps_480_samples': CP_TAPS.tolist(),
        'a_scale': a_scale, 'b_timing': b_timing, 'frames': rows,
        'fft_calibration_offsets_in_block': FFT_TRAIN_OFFSETS,
    }
    (out/'overlap_prefix_cancellation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
