"""Predict the target slot using the two other varying slots on withheld harmonics."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_slot_frame_power import DEGREES, SIGN_FLOOR, stable_signs


SEED = 20261121117
CONTROLS = 63


def evaluate_predictions(first, second, target, shifts):
    signs = stable_signs(np.stack((first, second)))
    count = int(np.count_nonzero(signs))
    if not count:
        raise ValueError('No predictions from the two donor slots')
    hard = (abs(target) >= SIGN_FLOOR) & (signs != 0)
    return {'eligible': count, 'mean_target_projection': float(signs @ target / count),
            'hard_witness_count': int(hard.sum()),
            'hard_witness_agrees': int(np.count_nonzero(signs[hard] * target[hard] > 0)),
            'rotations': [float(np.roll(signs, int(s)) @ target / count) for s in shifts]}


def main():
    require_run_context()
    hashes = {}
    first, second = [read(checked_artifact('v112-wav-slot-frame-power-' + ending, 'E114',
                                           'wav_slot_frame_power.json', hashes))
                     for ending in ('final', 'confirm')]
    if first != second:
        raise ValueError('Frozen per-frame runs differ')
    target = str(first['target_slot'])
    donors = [str(i) for i in first['observed_signs']['slot_order'] if str(i) != target]
    rng = np.random.default_rng(SEED)
    rows = []
    for row in first['windows']:
        shifts = rng.choice(np.arange(1, len(row['ordinals_in_window'])), size=CONTROLS, replace=False)
        fits = []
        for index, degree in enumerate(DEGREES):
            arms = []
            d0, d1, t = [np.asarray(row['targets'][slot][index]['real_projection_by_arm']) for slot in (*donors, target)]
            for arm in (0, 1):
                arms.append(evaluate_predictions(d0[arm], d1[arm], t[1 - arm], shifts))
            fits.append({'degree': degree, 'arms': arms})
        rows.append({'window_start_s': row['window_start_s'], 'time_group': row['time_group'],
                     'frame_rotations': shifts.tolist(), 'fits': fits})
    summary = []
    for index, degree in enumerate(DEGREES):
        directions = []
        for arm in (0, 1):
            rows_arm = [r['fits'][index]['arms'][arm] for r in rows]
            weights = [r['eligible'] for r in rows_arm]
            main_score = np.average([r['mean_target_projection'] for r in rows_arm], weights=weights)
            control = np.average([r['rotations'] for r in rows_arm], axis=0, weights=weights)
            directions.append({'eligible': sum(weights), 'target_projection': float(main_score),
                               'rotations_ge_main': int(np.count_nonzero(control >= main_score)),
                               'largest_rotation_projection': float(control.max()),
                               'hard_witness_count': sum(r['hard_witness_count'] for r in rows_arm),
                               'hard_witness_agrees': sum(r['hard_witness_agrees'] for r in rows_arm)})
        summary.append({'degree': degree, 'directions': directions})
    report = {'schema': 'starlink-wav-slot-sign-transfer/v1', 'seed': SEED, 'donor_slots': donors,
              'target_slot': target, 'input_sha256': {p.as_posix(): digest for p, digest in hashes.items()},
              'summary': summary, 'rows': rows}
    out = experiment_output('E114')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wav_slot_sign_transfer.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
