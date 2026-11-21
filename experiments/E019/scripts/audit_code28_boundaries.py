"""Compare Code 28 word boundaries without using any CRC or fitted FEC."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from code21_ring import evaluate, pack_blocks, recover, reduce_polynomial
from code28_core import LIFT, MOTHER_N, SYMBOLS, coordinates
from validate_code21 import digest
from validate_code37 import constellation_for

BACKEND = 'runs/v017-code28-hard/artifacts/experiments/E019/results'
PROJECTION_SEED = 2026112219


def last_symbol_indices(count, stride):
    words = np.arange(1, count + 1)
    if stride is None:
        return (words * MOTHER_N + 2) // 3 - 1
    return words * stride - 1


def fit_pair(labels):
    counts = np.bincount(labels, minlength=8)
    chosen = np.argsort(-counts, kind='stable')[:2]
    return sorted(chosen.tolist())


def positions_by_frame(labels, train_frames, indices, pair=None):
    subsets = {f: a[idx] for f, a in labels.items() if len(idx := indices(f, len(a)//SYMBOLS))}
    if pair is None:
        pair = fit_pair(np.concatenate([a for f, a in subsets.items() if f in train_frames]))
    return {'pair': pair,
            'frames': [{'frame': f, 'training': f in train_frames,
                        'hits': int(np.count_nonzero(np.isin(a, pair))), 'positions': len(a)} for f, a in subsets.items()]}


def max_projected_consensus(u, v, group, seed, trials=300):
    """Sample only calibration words, validate exact consensus in the small ring."""
    small = 16
    ub = [[reduce_polynomial(x, small) for x in row] for row in pack_blocks(u[group], LIFT)]
    vb = [[reduce_polynomial(x, small) for x in row] for row in pack_blocks(v[group], LIFT)]
    rng = np.random.default_rng(seed)
    best, invertible, distribution = 0, 0, {}
    for _ in range(trials):
        indices = rng.choice(len(ub), size=9, replace=False)
        fitted, _, _ = recover([ub[i] for i in indices], [vb[i] for i in indices], small)
        if fitted is None:
            continue
        invertible += 1
        predicted = evaluate(ub, fitted, small)
        score = sum(x == y for x, y in zip(predicted, vb, strict=True))
        best = max(best, score)
        distribution[str(score)] = distribution.get(str(score), 0) + 1
    return {'seed': seed, 'trials': trials, 'projection_lift': small, 'invertible_trials': invertible,
            'maximum_consensus_including_nine_fit_words': best, 'consensus_histogram': distribution}


def block_parity_projection(u, v, training, frames, seed, trials=199):
    """Exhaust every linear column on the nine block parities, with a transfer null."""
    x = (u.reshape(len(u), 9, LIFT).sum(axis=2) % 2).astype(np.uint8)
    y = (v.reshape(len(v), 7, LIFT).sum(axis=2) % 2).astype(np.uint8)
    all_columns = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
    candidate_predictions = (x @ all_columns.T) % 2

    def assess(labels, include_details=False):
        errors = np.count_nonzero(candidate_predictions[training, :, None] != labels[training, None, :], axis=0)
        best = errors.argmin(axis=0)
        holdout = candidate_predictions[~training][:, best]
        agreement = holdout == labels[~training]
        if include_details:
            return {'candidate_count_per_column': len(all_columns),
                    'tied_best_counts': np.count_nonzero(errors == errors.min(axis=0), axis=0).tolist(),
                    'training_disagreements_by_column': errors.min(axis=0).tolist(),
                    'holdout_agreement_by_column': agreement.sum(axis=0).tolist(),
                    'holdout_words': int((~training).sum()), 'selected_column_masks': best.tolist()}
        return int(agreement.sum())

    rng = np.random.default_rng(seed)
    result = assess(y, include_details=True)
    groups = [np.flatnonzero(frames == f) for f in np.unique(frames)]
    null = []
    for _ in range(trials):
        perm = np.arange(len(y))
        for group in groups:
            perm[group] = rng.permutation(group)
        null.append(assess(y[perm]))
    result.update(permutation_seed=seed, within_frame_whole_parity_word_maximum=int(max(null)),
                  within_frame_whole_parity_word_null=null)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    require_run_context()
    out = experiment_output('E019')
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT / BACKEND
    numeric = path / 'code28_observations.npz'
    metadata = path / 'code28.json'
    source = json.loads(metadata.read_text(encoding='utf-8'))
    with np.load(numeric, allow_pickle=False) as prior:
        words = prior['observations_with_neighbors'][:, 1:-1]
        hard = prior['hard_bits']
        groups = [prior['published_group'], prior['random_group']]
    frames = np.array([row['frame'] for row in source['words']])
    u, v, _ = coordinates(hard)
    points = constellation_for(*source['receiver_mapping'])
    label = hard.reshape(len(hard), SYMBOLS, 3) @ np.array([4, 2, 1])
    labels, captured = {}, {}
    for f in np.unique(frames):
        mask = frames == f
        meta = [r for r in source['words'] if r['frame'] == f]
        if any(b['start_resource'] - a['start_resource'] != SYMBOLS for a, b in zip(meta, meta[1:])):
            raise RuntimeError('Noncontiguous allocation')
        labels[int(f)] = label[mask].ravel()
        captured[int(f)] = words[mask].ravel()
    outcomes = []
    for group_name, group in zip(['published', 'randomized'], groups, strict=True):
        training = np.unique(frames[group]).tolist()
        separate = positions_by_frame(labels, training, lambda f, n: last_symbol_indices(n, SYMBOLS))
        pair = separate['pair']
        joint = positions_by_frame(labels, training, lambda f, n: last_symbol_indices(n, None), pair)
        joint_refitted = positions_by_frame(labels, training, lambda f, n: last_symbol_indices(n, None))
        offsets = {}
        for k in range(-8, 9):
            def shifted(f, n):
                idx = last_symbol_indices(n, SYMBOLS) + k
                return idx[(idx >= 0) & (idx < len(labels[f]))]
            offsets[str(k)] = positions_by_frame(labels, training, shifted, pair)
        # Omit the last two words so the count stays fixed for every stride.
        stride_scan = {}
        for stride in range(5400, 5501):
            stride_scan[str(stride)] = positions_by_frame(
                labels, training, lambda f, n: last_symbol_indices(n - 2, stride))
        final_margin = []
        for f, z in captured.items():
            ends = z[last_symbol_indices(len(z)//SYMBOLS, SYMBOLS)]
            e = abs(ends[:, None] - points[None, :]) ** 2
            dist_in = e[:, pair].min(axis=1)
            others = np.setdiff1d(np.arange(8), pair)
            margin = e[:, others].min(axis=1) - dist_in
            final_margin.append({'frame': f, 'min_distance_gap': float(margin.min()),
                                 'median_distance_gap': float(np.median(margin))})
        outcomes.append({'assignment': group_name, 'separate': separate, 'joint_fixed_pair': joint,
                         'joint_own_training_pair': joint_refitted, 'offset_controls': offsets,
                         'stride_scan_refit_pair_in_training_only': stride_scan,
                         'last_symbol_distance_margin': final_margin,
                         'projected_generator_screen': max_projected_consensus(u, v, group, PROJECTION_SEED),
                         'exhaustive_block_parity_projection': block_parity_projection(u, v, group, frames, PROJECTION_SEED + 1)})
    result = {'schema': 'starlink-code28-boundary/v1', 'input_npz_sha256': digest(numeric),
              'baseline_result_sha256': digest(metadata), 'received_words': len(words),
              'label_convention': source['receiver_mapping'], 'spare_label_hard_ones': hard[:, -2:].sum(axis=0).tolist(),
              'assignments': outcomes,
              'scope': 'Same published capture. A last-symbol pattern supports spacing. The failed finite projected fits do not exclude all encoders.'}
    (out / 'code28_boundaries.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for row in outcomes:
        print(row['assignment'], row['separate'], row['joint_fixed_pair'],
              row['projected_generator_screen'], flush=True)


if __name__ == '__main__':
    main()
