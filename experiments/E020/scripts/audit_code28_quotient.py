"""Measure how far a noisily observed Code 28 relation lifts beyond block parity."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from code28_core import coordinates
from quotient_lift_core import fit_layers, pascal_transform
from validate_code21 import digest

SOURCE_28 = 'runs/v017-code28-hard/artifacts/experiments/E019/results/code28_observations.npz'
META_28 = 'runs/v017-code28-hard/artifacts/experiments/E019/results/code28.json'
BASE_28 = 'runs/v017-code28-verified/artifacts/experiments/E019/results/code28_boundaries.json'
SOURCE_21 = 'runs/v014-code21-repeat/artifacts/experiments/E016/results/code21_observations.npz'
DEPTH = 16
TRIALS = 99
SEED = 2026112020


def transform(bits, width):
    return pascal_transform(bits.reshape(len(bits), -1, width))[..., :DEPTH]


def null_distribution(u, v, training, frames, seed):
    rng = np.random.default_rng(seed)
    keys = [np.flatnonzero(frames == f) for f in np.unique(frames)]
    outcomes = []
    for _ in range(TRIALS):
        shuffled = np.arange(len(frames))
        for key in keys:
            shuffled[key] = rng.permutation(key)
        _, report, _ = fit_layers(u, v[shuffled], training)
        outcomes.append([sum(row['holdout_agreement']) for row in report])
    return np.asarray(outcomes)


def main():
    require_run_context()
    out = experiment_output('E020')
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {p: digest(ROOT / p) for p in [SOURCE_28, META_28, BASE_28, SOURCE_21]}
    base = json.loads((ROOT / BASE_28).read_text(encoding='utf-8'))
    if (source_hashes[SOURCE_28] != base['input_npz_sha256']
            or source_hashes[META_28] != base['baseline_result_sha256']
            or source_hashes[SOURCE_21] != 'd5b781862d4289aea615621eb95f06a75e10239f8bb100442428974a98bf9ac0'):
        raise RuntimeError('Expected the archived reception evidence')
    with np.load(ROOT / SOURCE_28, allow_pickle=False) as data:
        u0, v0, _ = coordinates(data['hard_bits'])
        assignments = [('published', data['published_group']), ('randomized', data['random_group'])]
    frame_ids = np.array([w['frame'] for w in json.loads((ROOT / META_28).read_text(encoding='utf-8'))['words']])
    u, v = transform(u0, 1024), transform(v0, 1024)
    outputs, models = [], []
    arrays = {'u_t_coefficients': u, 'v_t_coefficients': v}
    for index, (name, training) in enumerate(assignments):
        model, rows, predicted = fit_layers(u, v, training)
        models.append(model)
        previous = base['assignments'][index]['exhaustive_block_parity_projection']
        if (rows[0]['selected_column_masks'] != previous['selected_column_masks']
                or rows[0]['holdout_agreement'] != previous['holdout_agreement_by_column']):
            raise RuntimeError('Constant term fails to reproduce E019')
        null = null_distribution(u, v, training, frame_ids, SEED + index)
        for degree, row in enumerate(rows):
            actual = sum(row['holdout_agreement'])
            row.update(holdout_agreement_total=actual, null_maximum=int(null[:, degree].max()),
                       null_exceedances=int(np.count_nonzero(null[:, degree] >= actual)))
        outputs.append({'name': name, 'training_frames': np.unique(frame_ids[training]).tolist(),
                        'holdout_frames': np.unique(frame_ids[~training]).tolist(), 'holdout_words': int((~training).sum()),
                        'layers': rows, 'null_seed': SEED + index})
        arrays.update({f'{name}_t_generator': model, f'{name}_predicted_parity': predicted,
                       f'{name}_whole_word_null': null, f'{name}_training': training})
        print(name, [r['holdout_agreement_total'] for r in rows], flush=True)
    agreed = np.all(models[0] == models[1], axis=(0, 1))
    first_disagreement = np.flatnonzero(~agreed)
    with np.load(ROOT / SOURCE_21, allow_pickle=False) as control:
        c = control['hard_bits']
        training = control['published_training_frames']
        p = control['parity_generator']
    if hashlib.sha256(p.tobytes()).hexdigest() != '65107672775f5d53fcf6f58f3390b26483c553320ef1fe1c0536768e49ac6c1b':
        raise RuntimeError('Code 21 expected matrix changed')
    p_t = pascal_transform(p[::512].reshape(10, 6, 512))[..., :DEPTH]
    control_fit, ctrl, _ = fit_layers(transform(c[:, :5120], 512), transform(c[:, 5120:], 512), training)
    result = {'schema': 'starlink-code28-quotient/v1', 'source_sha256': source_hashes, 'depth': DEPTH,
              'permutation_trials': TRIALS, 'assignments': outputs, 'equal_all_columns_by_degree': agreed.tolist(),
              'equal_prefix_terms': int(first_disagreement[0]) if len(first_disagreement) else DEPTH,
              'equal_columns_per_degree': np.all(models[0] == models[1], axis=0).sum(axis=0).tolist(),
              'code21_control': {'source_words': len(c), 'matched_coefficients': int(np.count_nonzero(control_fit == p_t)),
                                 'total_coefficients': p_t.size, 'exact_matrix': bool(np.array_equal(control_fit, p_t)),
                                 'layers': ctrl},
              'scope': 'Sequential estimates in a quotient of one capture. Prefix agreement does not certify a full matrix.'}
    print('Equal prefix terms', result['equal_prefix_terms'], 'Code 21 exact', result['code21_control']['exact_matrix'], flush=True)
    (out / 'code28_quotient.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'code28_quotient.npz', **arrays)


if __name__ == '__main__':
    main()
