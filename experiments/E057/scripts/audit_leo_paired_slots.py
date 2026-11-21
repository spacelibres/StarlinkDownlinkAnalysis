"""Cross-predict LEO IQ at matched OFDM resources after removing periodic structure."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, LEO_SNAPSHOT, experiment_output, require_run_context
from audit_leo_pairs_live import REPORT
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import read
from receiver_models import fit_frame, pn_reference


SEED = 2026112157
FRAMES = 14
GUARDS = (.6, 1.5)
SLOTS = np.arange(2, 302)
PILOT_IDS = np.arange(528, 536)


def partition_groups():
    order = np.random.default_rng(SEED).permutation(6)
    return np.sort(order[:3]), np.sort(order[3:])


def cycle_differences(values):
    """Expect frame x slot arrays, with optional feature dimensions after slot."""
    grouped = values[:, 30:].reshape(len(values), 6, 3, 15, *values.shape[2:])
    return np.diff(grouped, axis=2)


def subtract_pilots(values, design, train):
    calibration = design[train].reshape(-1, design.shape[-1])
    gain = np.linalg.lstsq(calibration, values[:, train].reshape(len(values), -1).T, rcond=None)[0]
    return values - np.einsum('gtsp,pf->fgts', design, gain)


def transfer(a, b, train, held):
    x = a[:, train].reshape(len(a), -1)
    y = b[:, train].reshape(len(b), -1)
    energy = np.sum(abs(y)**2, axis=1)
    gain = np.sum(y.conj()*x, axis=1)/np.maximum(energy, 1e-30)
    actual = a[:, held]
    predicted = b[:, held]*gain[:, None, None, None]
    norm = float(np.vdot(actual, actual).real)
    power = float(np.vdot(predicted, predicted).real)
    cross = float(np.vdot(actual, predicted).real)
    return (2*cross-power)/max(norm, 1e-30), cross/max(np.sqrt(norm*power), 1e-30)


def alignment_test(first, second, train, held):
    improvement, coherence = transfer(first, second, train, held)
    frame_mismatch = [transfer(first, np.roll(second, lag, axis=0), train, held)
                      for lag in range(1, len(first))]
    slot_mismatch = [transfer(first, np.roll(second, lag, axis=-1), train, held)
                     for lag in range(1, 15)]
    controls = frame_mismatch + slot_mismatch
    supported = improvement > max(0., max(r[0] for r in controls))
    supported &= coherence > max(0., max(r[1] for r in controls))
    return {
        'paired_improvement': improvement, 'paired_signed_coherence': coherence,
        'wrong_frame_by_roll': frame_mismatch, 'wrong_slot_class_by_roll': slot_mismatch,
        'passes_all_comparisons': bool(supported),
    }


def evaluate_tone(a, b, k, pilots, train, held):
    is_control = k in PILOT_IDS
    reference = np.ones(len(SLOTS)) if is_control else pn_reference(k, SLOTS)
    a = cycle_differences((a*reference.conj()[None, :]))
    b = cycle_differences((b*reference.conj()[None, :]))
    known = pilots[:, PILOT_IDS != k]
    # The first two cycles are discarded so the rolled pilot at the start does
    # not affect the evaluated support. The absent next symbol at the end is zero.
    previous = np.vstack((np.zeros((1, known.shape[1])), known[:-1]))
    following = np.vstack((known[1:], np.zeros((1, known.shape[1]))))
    spill = np.column_stack((previous, known, following))*reference.conj()[:, None]
    design = cycle_differences(spill[None, ...])[0]
    tested = {
        'carrier': k, 'known_pilot_control': bool(is_control),
        'after_lag_15_difference': alignment_test(a, b, train, held),
        'also_remove_pilot_leakage': alignment_test(
            subtract_pilots(a, design, train), subtract_pilots(b, design, train), train, held
        ),
    }
    if is_control:
        exact = cycle_differences(pilots[:, k-528][None, :])
        exact = np.broadcast_to(exact, a.shape)
        tested['known_reference_to_each_rx'] = [transfer(rx, exact, train, held) for rx in (a, b)]
    return tested


def extract(iq, anchor, pilots, training_slots, rate, guard, *, withheld_pilot=None):
    rows = [
        fit_frame(iq, anchor, frame, pilots, training_slots, rate, guard, withheld_pilot=withheld_pilot)
        for frame in range(FRAMES)
    ]
    if any(row is None for row in rows):
        raise ValueError('Incomplete aligned frame in paired input')
    carriers = rows[0]['unknown_indices']
    if any(not np.array_equal(row['unknown_indices'], carriers) for row in rows):
        raise ValueError('Projected carriers differ over frames')
    values = np.stack([row['unknown'] for row in rows], axis=2)
    return {
        'symbols': {int(k): values[:, i, :].T for i, k in enumerate(carriers)},
        'available': rows[0]['indices'].tolist(),
        'residual_cfo_hz': [row['residual_cfo_hz'] for row in rows],
        'cfo_boundary_frames': [i for i, row in enumerate(rows) if row['fit_at_search_boundary']],
    }


def main():
    require_run_context()
    from audit_nonpilot_leakage import expected

    source = EXTERNAL_LEO_ROOT/REPORT/'paired_RX'
    snapshot = LEO_SNAPSHOT/REPORT/'paired_RX'
    files = ('selection.json', 'selected_iq.npz')
    inputs = {name: sha256(source/name) for name in (*files, 'differential_chirp.json')}
    if any(inputs[name] != sha256(snapshot/name) for name in files):
        raise ValueError('Paired IQ or selection differs relative to preserved evidence')
    selection = read(source/'selection.json')
    rate = selection['sample_rate_hz']
    train, held = partition_groups()
    mask = np.zeros(20, dtype=bool)
    mask[2+(3*train[:, None]+np.arange(3)).ravel()] = True
    training_slots = np.repeat(mask, 15)
    pairs = []
    with np.load(source/'selected_iq.npz', allow_pickle=False) as data:
        for n, lag in enumerate(read(source/'differential_chirp.json')):
            first, second = selection['windows'][2*n:2*n+2]
            fields = ('actual_rf_hz', 'visit_index', 'valid_start_counter', 'edge')
            if any(first['probe'][k] != second['probe'][k] for k in fields):
                raise ValueError('Paired observations lack a common sampling record')
            if [first['observation_id'], second['observation_id']] != lag['observation_ids']:
                raise ValueError('Paired observations do not match the delay record')
            anchors = [row['winner'] for row in (first, second)]
            epochs = [r['integer_epoch_sample']+r['fractional_epoch_offset_samples'] for r in anchors]
            offset = epochs[1]-epochs[0]
            if abs(offset-lag['delay_samples']) > .1 or abs(offset) > rate/750/2:
                raise ValueError('Archived delays do not support same-index frame pairing')
            iq = [data[f'iq_{2*n+i}'] for i in (0, 1)]
            rows = []
            for guard in GUARDS:
                a, b = [extract(y, c, expected, training_slots, rate, guard)
                        for y, c in zip(iq, anchors, strict=True)]
                available = [sorted(x['symbols']) for x in (a, b)]
                common = sorted(set(available[0]) & set(available[1]))
                stats = [
                    evaluate_tone(a['symbols'][k], b['symbols'][k], k, expected, train, held) for k in common
                ]
                common_pilots = sorted(set(a['available']) & set(b['available']) & set(PILOT_IDS.tolist()))
                controls = []
                for k in common_pilots:
                    ca, cb = [extract(y, c, expected, training_slots, rate, guard, withheld_pilot=k)
                              for y, c in zip(iq, anchors, strict=True)]
                    controls.append(evaluate_tone(ca['symbols'][k], cb['symbols'][k], k, expected, train, held))
                rows.append({
                    'guard_samples': guard, 'individual_nonpilot_carriers': available,
                    'common_nonpilot_carriers': common, 'nonpilot_tests': stats, 'pilot_positives': controls,
                    'residual_cfo_hz': [x['residual_cfo_hz'] for x in (a, b)],
                    'cfo_boundary_frames': [x['cfo_boundary_frames'] for x in (a, b)],
                })
            pairs.append({
                'time_s': lag['time_s'], 'epoch_difference_samples': offset,
                'observation_ids': lag['observation_ids'], 'source_coords': {k: first['probe'][k] for k in fields},
                'paths': rows,
            })
            print(f"Paired window at {lag['time_s']:.3f} s, common nonpilots: {rows[0]['common_nonpilot_carriers']}",
                  flush=True)
    source_template = LEO_SNAPSHOT/'src/leo/analysis/starlink/templates.py'
    result = {
        'schema': 'starlink-leo-paired-slot-transfer/v2', 'seed': SEED, 'sample_rate_hz': rate,
        'external_directory': str(source), 'external_inputs_sha256': inputs,
        'snapshot_pilot_template_sha256': sha256(source_template),
        'training_cycle_groups': train.tolist(), 'held_cycle_groups': held.tolist(),
        'cycles_by_group': np.arange(2, 20).reshape(6, 3).tolist(), 'pairs': pairs,
    }
    out = experiment_output('E057')
    out.mkdir(parents=True, exist_ok=True)
    (out/'paired_slot_transfer.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
