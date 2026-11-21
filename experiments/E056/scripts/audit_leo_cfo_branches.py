"""Check archived LEO carrier branches using unseen samples within OFDM symbols."""

import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, LEO_SNAPSHOT, experiment_output, require_run_context
from audit_leo_pairs_live import BLOCKS, REPORT
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import read


FINE_HZ = np.arange(-400., 401., 50.)
NAMES = ('archived', 'slot_minus_one', 'slot_plus_one', 'slot_minus_two',
         'slot_plus_two', 'tone_minus_one', 'tone_plus_one', 'rolled_17')


def aperture_scores(basis, y, time, early, frequencies):
    """Return held-out energy improvement over predicting zero, in both directions."""
    corrected = y[:, None] * np.exp(-2j*np.pi*time[:, None]*frequencies)
    improvement = np.zeros(len(frequencies))
    for training in (early, ~early):
        design = basis[training]
        gains = np.linalg.lstsq(design, corrected[training], rcond=None)[0]
        predicted = basis[~training] @ gains
        improvement += np.sum(
            2*(corrected[~training].conj()*predicted).real - abs(predicted)**2, axis=0
        )
    return improvement, float(np.vdot(y, y).real)


def score_splits(improvements, energy, names):
    rows = []
    for training_blocks in combinations(range(len(BLOCKS)), 2):
        train = np.array([frame for i in training_blocks for frame in BLOCKS[i]])
        held = np.setdiff1d(np.arange(len(energy)), train)
        sums = improvements[train].sum(axis=0)
        best = sums.argmax(axis=1)
        rows.append({
            'training_blocks': list(training_blocks),
            'held_frames': held.tolist(),
            'models': {
                name: {
                    'grid_index': int(best[i]),
                    'training_improvement': float(sums[i, best[i]]/energy[train].sum()),
                    'held_improvement': float(improvements[held, i, best[i]].sum()/energy[held].sum()),
                }
                for i, name in enumerate(names)
            },
        })
    return rows


def frame_scores(iq, rate, epoch, cfo, edge, frame_clock, templates):
    slot_s = templates.OFDM_SYMBOL_DURATION_S
    prefix_s = templates.CYCLIC_PREFIX_DURATION_S
    tone_hz = templates.SUBCARRIER_SPACING_HZ
    frame_hz = templates.FRAME_RATE_HZ
    centers = np.array([0., -1/slot_s, 1/slot_s, -2/slot_s, 2/slot_s, -tone_hz, tone_hz, 0.])
    grid = centers[:, None] + FINE_HZ
    improvements, energies, coverage = [], [], []
    for frame in range(sum(map(len, BLOCKS))):
        origin = epoch + frame*rate/frame_hz
        model = frame_clock(iq, rate, origin, cfo, edge)
        rolled = frame_clock(iq, rate, origin, cfo, edge, roll=17)
        # Recover the intra-symbol coordinate of the externally guarded samples.
        first = np.ceil(origin + (2*slot_s+prefix_s)*rate + .6)
        time = model.time - model.time[0] + (first-origin)/rate
        local = np.remainder(time, slot_s)
        if local.min() < prefix_s+.6/rate-1e-12 or local.max() > slot_s-.6/rate+1e-12:
            raise ValueError('Sample support differs from the original FrameClock guard')
        early = local < (prefix_s+slot_s)/2
        exact, energy = aperture_scores(model.basis, model.values, model.time, early, grid[:-1].ravel())
        control, _ = aperture_scores(rolled.basis, rolled.values, rolled.time, early, grid[-1])
        improvements.append(np.concatenate((exact, control)).reshape(grid.shape))
        energies.append(energy)
        coverage.append([int(early.sum()), int((~early).sum())])
    return np.array(improvements), np.array(energies), coverage, grid


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    files = [REPORT/folder/name for folder in ('DSX01-S05', 'paired_RX')
             for name in ('selection.json', 'selected_iq.npz')]
    inputs = {}
    for rel in files:
        inputs[rel.as_posix()] = sha256(EXTERNAL_LEO_ROOT/rel)
        if inputs[rel.as_posix()] != sha256(LEO_SNAPSHOT/rel):
            raise ValueError('External IQ or metadata differs from the preserved input')
    for rel in ('src/leo/analysis/starlink/templates.py', 'src/leo/contracts/states.py',
                'tools/dsx01_pilot_clock.py', (REPORT/'paired_RX/differential_chirp.json').as_posix()):
        inputs[rel] = sha256(EXTERNAL_LEO_ROOT/rel)
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT/'src'), str(EXTERNAL_LEO_ROOT/'tools')]
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    groups, aliases, iq_digests = {}, {}, {}
    for folder in ('DSX01-S05', 'paired_RX'):
        source = EXTERNAL_LEO_ROOT/REPORT/folder
        selection = read(source/'selection.json')
        rate = selection['sample_rate_hz']
        with np.load(source/'selected_iq.npz', allow_pickle=False) as archive:
            for entry in selection['windows']:
                key = f"{folder}/iq_{entry['number']}"
                iq = archive[f"iq_{entry['number']}"]
                observation = entry['observation_id']
                winner = entry['winner']
                probe = entry['probe']
                epoch = winner['integer_epoch_sample'] + winner['fractional_epoch_offset_samples']
                cfo = winner['fractional_tracking_cfo_hz']
                digest = hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest()
                if digest in iq_digests:
                    matching = groups[iq_digests[digest]]
                    if (observation, epoch, cfo) != (matching['observation_id'], matching['epoch_sample'],
                                                     matching['original_cfo_hz']):
                        raise ValueError('Duplicate IQ has conflicting coordinates')
                    aliases[key] = iq_digests[digest]
                    continue
                iq_digests[digest] = key
                improvements, energy, coverage, grid = frame_scores(
                    iq, rate, epoch, cfo, probe['edge'], FrameClock, templates
                )
                groups[key] = {
                    'observation_id': observation, 'iq_sha256_complex64_le': digest,
                    'edge': probe['edge'], 'sample_rate_hz': rate, 'epoch_sample': epoch,
                    'original_cfo_hz': cfo, 'rx': entry['receiver'],
                    'context_time_s': entry['context_time_s'],
                    'original_candidate_offsets_hz': [
                        c['fractional_tracking_cfo_hz']-cfo for c in probe['candidates']
                    ],
                    'frame_aperture_sample_counts': coverage, 'frame_energy': energy.tolist(),
                    'frame_held_energy_improvements': improvements.tolist(),
                    'splits': score_splits(improvements, energy, NAMES),
                }
                print(key, 'measured', flush=True)
    pairs = []
    for i, old in enumerate(read(EXTERNAL_LEO_ROOT/REPORT/'paired_RX/differential_chirp.json')):
        key_a, key_b = f'paired_RX/iq_{2*i}', f'paired_RX/iq_{2*i+1}'
        a, b = groups[aliases.get(key_a, key_a)], groups[key_b]
        if [a['observation_id'], b['observation_id']] != old['observation_ids']:
            raise ValueError('Paired observations differ from the original cross-IQ diagnostics')
        pairs.append({
            'pair_time_s': old['time_s'], 'first_input': key_a, 'second_input': key_b,
            'archived_cross_iq_hz': old['template_free_cross_IQ']['frequency_difference_hz'],
            'archived_pilot_difference_hz': a['original_cfo_hz']-b['original_cfo_hz'],
            'held_grid_differences_hz': [
                a['original_cfo_hz']-b['original_cfo_hz'] +
                FINE_HZ[x['models']['archived']['grid_index']] -
                FINE_HZ[y['models']['archived']['grid_index']]
                for x, y in zip(a['splits'], b['splits'], strict=True)
            ],
        })
    result = {
        'schema': 'starlink-leo-cfo-aperture/v1', 'external_root': str(EXTERNAL_LEO_ROOT),
        'external_inputs_sha256': inputs, 'blocks': BLOCKS, 'model_order': NAMES,
        'grid_offsets_hz': grid.tolist(), 'duplicate_iq': aliases, 'windows': groups, 'pairs': pairs,
    }
    out = experiment_output('E056')
    out.mkdir(parents=True, exist_ok=True)
    (out/'leo_cfo_aperture.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
