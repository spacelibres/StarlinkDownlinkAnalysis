"""Use joint QAM parity likelihoods to test projected quasi-cyclic constraints."""

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


SEED = 2026112160
LIFTS = (128, 256, 512, 1024)
MIN_MAGNITUDE = .1
TEMPERATURES = np.arange(12, 91, 3)*.001
PERMUTATIONS = 199
SEARCH_BATCH = 4096


def qam_parity_moments(values, points, labels, variance):
    """Marginalize the XOR of all bits jointly over the constellation."""
    sign = 1-2*(np.asarray(labels).sum(axis=1) % 2).astype(float)
    distances = abs(np.asarray(values)[..., None]-points)**2
    distances -= distances.min(axis=-1, keepdims=True)
    weights = np.exp(-distances/variance)
    return ((weights @ sign)/weights.sum(axis=-1)).clip(-1, 1)


def search_masks(width, depth):
    return [cols for n in range(1, depth+1) for cols in combinations(range(width), n)]


def fit_projection(u, v, train, masks):
    """Search sparse systematic-block products using only training codewords."""
    indices = np.full((len(masks), max(map(len, masks))), u.shape[1], dtype=int)
    for i, cols in enumerate(masks):
        indices[i, :len(cols)] = cols
    augmented = np.column_stack((u, np.ones(len(u))))
    held_outputs = np.arange(v.shape[1])
    best = np.zeros(v.shape[1], dtype=int)
    score = np.zeros(v.shape[1])
    train_u, train_v = augmented[train], v[train]
    for start in range(0, len(indices), SEARCH_BATCH):
        batch = indices[start:start+SEARCH_BATCH]
        candidate = np.prod(train_u[:, batch], axis=2)
        statistic = candidate.T @ train_v
        local = abs(statistic).argmax(axis=0)
        local_score = statistic[local, held_outputs]
        better = abs(local_score) > abs(score)
        best[better] = start+local[better]
        score[better] = local_score[better]
    signs = np.where(score >= 0, 1, -1)
    out = np.prod(augmented[:, indices[best]], axis=2)*signs
    rules = [{'system_columns': list(masks[i]), 'sign': int(s), 'train_signed_sum_after_sign': float(abs(weight))}
             for i, s, weight in zip(best, signs, score, strict=True)]
    return rules, out


def folded(values, width):
    return np.prod(values.reshape(len(values), -1, width), axis=-1)


def validation_orders(frames, rng):
    groups = [np.flatnonzero(frames == frame) for frame in np.unique(frames)]
    orders = [np.arange(len(frames))]
    for _ in range(PERMUTATIONS):
        order = np.arange(len(frames))
        for group in groups:
            if len(group) > 1:
                order[group] = np.roll(group, rng.integers(1, len(group)))
        orders.append(order)
    return np.stack(orders)


def validation_scores(prediction, output, orders):
    paired = (prediction[None, ...]*output[orders]).clip(-.999, .999)
    return np.log1p(paired).mean(axis=(1, 2))


def reliability(values):
    evidence = abs(values)
    return {
        'count': int(evidence.size),
        'absolute_mean': float(evidence.mean()),
        'absolute_median': float(np.median(evidence)),
        'count_at_least_point_one': int(np.count_nonzero(evidence >= MIN_MAGNITUDE)),
    }


def delimit_known(pn_record, code, symbol_values, reference):
    blocks = []
    for word in pn_record['words']:
        if word['format'] != code or not word['match']['accepted']:
            continue
        match = word['match']
        first = (match['start']+3)//4
        last = min(match['stop'], SYSTEMATIC_ENDS[code])//4
        if first >= last:
            raise ValueError('Known PN has no complete 16QAM symbol')
        bits = reference[4*first:4*last].reshape(-1, 4)
        truth = 1-2*(bits.sum(axis=1) % 2).astype(float)
        blocks.append((word['frame'], word['candidate_index'], first, last, truth,
                       symbol_values[word['candidate_index'], first:last]))
    return blocks


def calibrate(words, points, labels, train_frames):
    values = np.concatenate([r[-1] for r in words if r[0] in train_frames])
    truth = np.concatenate([r[-2] for r in words if r[0] in train_frames])
    losses = [-np.mean(np.log1p((qam_parity_moments(values, points, labels, t)*truth).clip(-.999, .999)))
              for t in TEMPERATURES]
    return {'variance': float(TEMPERATURES[np.argmin(losses)]), 'known_symbols_used': len(truth),
            'calibration_losses': [float(x) for x in losses]}


def pn_holdout(known, train_frames, soft, hard):
    rows = {}
    for lift in LIFTS:
        q = lift//4
        reference, posterior, nearest = [], [], []
        for frame, word, start, stop, truth, _ in known:
            if frame in train_frames:
                continue
            first, last = (start+q-1)//q, stop//q
            if first < last:
                reference.extend(truth[first*q-start:last*q-start].reshape(-1, q).prod(axis=1))
                posterior.extend(soft[word, first*q:last*q].reshape(-1, q).prod(axis=1))
                nearest.extend(hard[word, first*q:last*q].reshape(-1, q).prod(axis=1))
        answer = np.asarray(reference)
        moments = np.asarray(posterior)
        gate = abs(moments) >= MIN_MAGNITUDE
        rows[lift] = {'folds': len(answer), 'hard_correct': int(np.count_nonzero(answer == nearest)),
                      'high_confidence_correct': int(np.count_nonzero((answer == np.sign(moments)) & gate)),
                      'posterior': reliability(moments)}
    return rows


def run_16qam(archive, pn_record, train_frames):
    points, labels = archive['constellation_points'], archive['frozen_gray_mapping']
    reference = equality_predictions(np.arange(16384), pn_bits(), pn_record['fit']['winner'])
    data, tested = {}, []
    for code in (66, 70):
        frames = archive[f'mcs{code}_frames']
        train = np.isin(frames, train_frames)
        orders = validation_orders(frames[~train], np.random.default_rng(SEED+code))
        refined = archive[f'mcs{code}_refined'][:, 1:-1]
        known = delimit_known(pn_record, code, refined, reference)
        calibration = calibrate(known, points, labels, train_frames)
        cutoff = SYSTEMATIC_ENDS[code]
        variants = {}
        point_parity = 1-2*(labels.sum(axis=1) % 2).astype(float)
        for name in ('refined', 'baseline'):
            y = archive[f'mcs{code}_{name}'][:, 1:-1]
            batches = np.array_split(y, 12)
            soft = np.vstack([qam_parity_moments(b, points, labels, calibration['variance']) for b in batches])
            hard = np.vstack([point_parity[np.argmin(abs(b[..., None]-points), axis=-1)] for b in batches])
            variants[name] = pn_holdout(known, train_frames, soft, hard)
            for lift in LIFTS:
                if cutoff % lift:
                    continue
                projected = folded(soft, lift//4)
                u, v = np.hsplit(projected, [cutoff//lift])
                masks = search_masks(u.shape[1], 3)
                rules, predict = fit_projection(u, v, train, masks)
                scores = validation_scores(predict[~train], v[~train], orders)
                top_four = np.partition(abs(u[~train]), -4, axis=1)[:, -4:].prod(axis=1)
                tested.append({
                    'mcs': code, 'frontend': name, 'lift': lift, 'fit_words': int(train.sum()),
                    'held_words': int((~train).sum()), 'candidate_masks_per_output': len(masks),
                    'rules': rules, 'system_soft_fold_held': reliability(u[~train]),
                    'parity_soft_fold_held': reliability(v[~train]), 'held_joint_moments': reliability(
                        predict[~train]*v[~train]
                    ),
                    'held_four_or_more_joint_magnitude_upper_bound': reliability(top_four[:, None]*v[~train]),
                    'identity_and_controls': scores.tolist(),
                })
        data[code] = {'calibration': calibration, 'known_pn_heldout': variants}
    score = np.array([row['identity_and_controls'] for row in tested])
    null = score[:, 1:]
    scaled = (score-null.mean(axis=1, keepdims=True))/np.maximum(null.std(axis=1, keepdims=True), 1e-30)
    joint = scaled[:, 1:].max(axis=0)
    for i, row in enumerate(tested):
        row['standardized_held_score'] = float(scaled[i, 0])
        row['fraction_family_controls_at_least_observed'] = (1+int(np.count_nonzero(joint >= scaled[i, 0])))/(PERMUTATIONS+1)
    return data, tested


def code5_control(source, report, generator):
    words = source['decoded_code5_words']
    y = source['observed_code5_symbols']
    groups = np.array([r['candidate_index'] for r in report['words']])
    eligible = np.array([r['decoder']['converged'] for r in report['words']], dtype=bool)
    if not eligible.all() or not np.array_equal(words[:, 1024:], words[:, :1024] @ generator % 2):
        raise ValueError('Code 5 reference words do not satisfy the frozen generator')
    rng = np.random.default_rng(SEED+5)
    frames = rng.permutation(np.unique(groups))
    train = np.isin(groups, frames[:len(frames)//2])
    known = 1-2*words.astype(float)
    variance = float(np.mean(abs(y[train]-known[train])**2))
    moments = folded(np.tanh(2*y.real/variance), 128)
    truth = folded(known, 128)
    matrix = generator.reshape(1024, 8, 128).sum(axis=2) % 2
    if not np.all(matrix.reshape(8, 128, 8) == matrix[::128, None, :]):
        raise ValueError('Code 5 block fold depends on bit offset within the block')
    mask = matrix[::128].T.astype(bool)
    actual = np.stack([np.prod(truth[:, :8][:, a], axis=1) for a in mask], axis=1)
    np.testing.assert_array_equal(actual, truth[:, 8:])
    controls = validation_orders(groups[~train], rng)
    fixed_prediction = np.stack([np.prod(moments[:, :8][:, a], axis=1) for a in mask], axis=1)
    fit, predicted = fit_projection(moments[:, :8], moments[:, 8:], train, search_masks(8, 8))
    return {
        'codewords': len(words), 'calibration_frames': sorted(np.unique(groups[train]).tolist()),
        'held_frames': sorted(np.unique(groups[~train]).tolist()), 'variance': variance,
        'actual_system_masks': [np.flatnonzero(a).tolist() for a in mask], 'fitted_rules': fit,
        'held_projection_powers': reliability(moments[~train]),
        'observed_held_fold_disagreements': int(np.count_nonzero(np.sign(moments[~train]) != truth[~train])),
        'fixed_held_scores': validation_scores(fixed_prediction[~train], moments[~train, 8:], controls).tolist(),
        'fitted_held_scores': validation_scores(predicted[~train], moments[~train, 8:], controls).tolist(),
    }


def main():
    require_run_context()
    inputs = {}
    def evidence(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    source16 = 'v026-ut-qam-extract-repeat'
    pn = read(evidence('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json'))
    split = read(evidence(source16, 'E028', 'ut_qam_extraction.json'))
    with np.load(evidence(source16, 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as archive:
        calibration, targets = run_16qam(archive, pn, split['calibration_frames'])
    wide = read(evidence('v022-ut-replay', 'E014', 'ut_transfer.json'))
    with np.load(evidence('v010-code5-parity-02', 'E012', 'code5_observations.npz'), allow_pickle=False) as saved:
        generator = saved['parity_generator']
    with np.load(evidence('v022-ut-replay', 'E014', 'ut_observations.npz'), allow_pickle=False) as saved:
        control = code5_control(saved, wide, generator)
    out = experiment_output('E060')
    out.mkdir(parents=True, exist_ok=True)
    data = {
        'schema': 'starlink-qc-soft-folds/v1', 'seed': SEED, 'permutations': PERMUTATIONS,
        'inputs_sha256': {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        'reliability_threshold': MIN_MAGNITUDE, 'temperature_grid': TEMPERATURES.tolist(),
        'training_frames': split['calibration_frames'], 'held_frames': split['reserved_frames'],
        'calibration_and_pn_holdout': calibration, 'target_cases': targets, 'code5_positive': control,
    }
    (out/'qc_soft_folds.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    for row in targets:
        print(row['mcs'], row['frontend'], row['lift'], row['fraction_family_controls_at_least_observed'], flush=True)


if __name__ == '__main__':
    main()
