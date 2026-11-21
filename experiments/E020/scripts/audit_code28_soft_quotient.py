"""Try joint-label reliability in the Code 28 quotient without fitting holdouts."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code28_quotient import BASE_28, DEPTH, META_28, SOURCE_28, transform
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYSTEMATIC, coordinates
from quotient_lift_core import fit_layers
from soft_quotient_core import expected_check_agreement, label_parity_means, mask_symbols
from validate_code21 import digest
from validate_code37 import constellation_for

NOISE_VARIANCES = [.01, .02, .04]
WEIGHT_VARIANCES = [.015, .025, .04]
WEIGHT_POWERS = [.5, 1., 2., 3.]
SEED = 2026112120
PERMUTATIONS = 199


def coefficient_fit(means, u, v, training):
    options = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
    shifts = np.arange(LIFT)
    terms = np.array([((shifts & d) == d) for d in range(DEPTH)], dtype=np.uint8)
    toggles = []
    for b in range(9):
        mask = np.zeros(MOTHER_N + 2, dtype=np.uint8)
        start = max(b * LIFT - OMITTED, 0)
        stop = (b+1) * LIFT - OMITTED
        mask[start:stop] = 1
        toggles.append(mask_symbols(mask))
    toggles = np.stack(toggles)
    model = np.empty((9, 7, DEPTH), dtype=np.uint8)
    prediction = np.zeros((len(u), DEPTH, 7), dtype=np.uint8)
    u_coeff = u.transpose(0, 2, 1)
    audits = []
    for d in range(DEPTH):
        selected, gaps, ties, mean_selected = [], [], [], []
        for j in range(7):
            pattern = np.zeros(MOTHER_N + 2, dtype=np.uint8)
            if d:
                source_mask = (model[:, j, :d] @ terms[d:0:-1]) % 2
                pattern[:SYSTEMATIC] = source_mask.ravel()[OMITTED:]
            start = MOTHER_K + j*LIFT
            pattern[start:start+LIFT] = terms[d]
            products = expected_check_agreement(means, mask_symbols(pattern), toggles, options)
            score = products.sum(axis=0)
            best = int(score.argmax())
            second = np.partition(score, -2)[-2]
            selected.append(best)
            gaps.append(float(score[best]-second))
            ties.append(int(np.count_nonzero(score==score[best])))
            mean_selected.append(float(score[best]/len(means)))
            model[:, j, d] = options[best]
        prediction[:, d:] ^= (u_coeff[:, :DEPTH-d] @ model[:, :, d]) % 2
        agreement = prediction[~training, d] == v[~training, :, d]
        audits.append({'degree': d, 'selected_column_masks': selected, 'score_margin': gaps,
                       'tied_scores': ties, 'mean_training_signed_score': mean_selected,
                       'holdout_agreement_by_column': agreement.sum(axis=0).tolist(),
                       'holdout_agreement_total': int(agreement.sum())})
    return model, prediction.transpose(0, 2, 1), audits


def transfer_null(predicted, target, frames, seed):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(frames == f) for f in np.unique(frames)]
    count = []
    for _ in range(PERMUTATIONS):
        selected = np.arange(len(target))
        for group in groups:
            selected[group] = rng.permutation(group)
        count.append(np.count_nonzero(predicted==target[selected], axis=(0, 1)))
    return np.stack(count)


def word_weight_bank(distances, u, v, groups, frames):
    """Record the complete exploratory bank rather than selecting a winner."""
    cases, arrays = [], {}
    for variance in WEIGHT_VARIANCES:
        wrong_symbols = (1 - 1 / np.exp(-distances/variance).sum(axis=-1)).sum(axis=1)
        for power in WEIGHT_POWERS:
            weights = (1 + wrong_symbols)**-power
            estimates, forecasts = [], []
            for index, group in enumerate(groups):
                model, rows, predicted = fit_layers(u, v, group, word_weights=weights)
                expected = v[~group]
                null = transfer_null(predicted[~group], expected, frames[~group], SEED + index)
                estimates.append(model)
                case = len(cases)
                arrays[f'weighted_case_{case}_group_{index}_coefficients'] = model
                arrays[f'weighted_case_{case}_group_{index}_null'] = null
                forecasts.append({'holdout_words': int((~group).sum()),
                                  'holdout_first_order': sum(rows[1]['holdout_agreement']),
                                  'fixed_null_maximum_first_order': int(null[:, 1].max()),
                                  'layers': rows})
            matching = np.all(estimates[0] == estimates[1], axis=0).sum(axis=0)
            cases.append({'noise_variance_assumption': variance, 'weight_exponent': power,
                          'matching_columns_per_degree': matching.tolist(), 'assignments': forecasts})
    return cases, arrays


def main():
    require_run_context()
    out = experiment_output('E020')
    out.mkdir(parents=True, exist_ok=True)
    source = {r: digest(ROOT / r) for r in [SOURCE_28, META_28, BASE_28]}
    e019 = json.loads((ROOT / BASE_28).read_text(encoding='utf-8'))
    if source[SOURCE_28] != e019['input_npz_sha256'] or source[META_28] != e019['baseline_result_sha256']:
        raise RuntimeError('Archived observation changed')
    with np.load(ROOT / SOURCE_28, allow_pickle=False) as data:
        samples = data['observations_with_neighbors'][:, 1:-1]
        u0, v0, _ = coordinates(data['hard_bits'])
        assignments = [data['published_group'], data['random_group']]
    frames = np.array([w['frame'] for w in json.loads((ROOT / META_28).read_text(encoding='utf-8'))['words']])
    u, v = transform(u0, LIFT), transform(v0, LIFT)
    distances = abs(samples[..., None]-constellation_for(2, False, True))**2
    distances -= distances.min(axis=-1, keepdims=True)
    report, arrays = [], {}
    for model_no, noise in enumerate(NOISE_VARIANCES):
        fact = np.exp(-distances/noise)
        groups, matrices = [], []
        for i, training in enumerate(assignments):
            posterior = fact[training] / fact[training].sum(axis=-1, keepdims=True)
            means = label_parity_means(posterior)
            p, predicted, rows = coefficient_fit(means, u, v, training)
            matrices.append(p)
            null = transfer_null(predicted[~training], v[~training], frames[~training], SEED + i)
            for j, row in enumerate(rows):
                score = row['holdout_agreement_total']
                row.update(null_maximum=int(null[:, j].max()), null_exceedances=int(np.count_nonzero(null[:, j]>=score)))
            groups.append({'training_frames': np.unique(frames[training]).tolist(),
                           'reserved_frames': np.unique(frames[~training]).tolist(),
                           'reserved_words': int((~training).sum()), 'null_seed': SEED + i, 'layers': rows})
            arrays[f'variance_{model_no}_group_{i}_coefficients'] = p
            arrays[f'variance_{model_no}_group_{i}_reserved_prediction'] = predicted[~training]
            arrays[f'variance_{model_no}_group_{i}_null'] = null
        same = np.all(matrices[0] == matrices[1], axis=0).sum(axis=0)
        unequal = np.flatnonzero(same != 7)
        report.append({'noise_variance_assumption': noise, 'coefficient_agreement_columns': same.tolist(),
                       'entire_equal_prefix_terms': int(unequal[0]) if len(unequal) else DEPTH,
                       'assignments': groups})
        print(noise, 'same columns', same.tolist(),
              'heldout first four', [[r['holdout_agreement_total'] for r in g['layers'][:4]] for g in groups], flush=True)
    weighted, weights_arrays = word_weight_bank(distances, u, v, assignments, frames)
    arrays.update(weights_arrays)
    result = {'schema': 'starlink-code28-soft-quotient/v1', 'source_sha256': source, 'depth': DEPTH,
              'models': report, 'word_weighted_models': weighted, 'transfer_null_trials': PERMUTATIONS,
              'scope': 'Follow-up after hard-decision fitting. Fixed-model permutation test only. Distance likelihoods approximate conditional independence.'}
    (out / 'code28_soft_quotient.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'code28_soft_quotient.npz', **arrays)


if __name__ == '__main__':
    main()
