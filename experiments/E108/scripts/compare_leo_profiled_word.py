"""Compare the completed blind replays with a separately frozen wideband word."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_leo_blind_word import best_equivalent_forms, cyclic_distance
from audit_ut_repeated_data import checked_artifact, read


def consensus_comparison(anchor, consensus, reference):
    keep = consensus['retained_positions']
    observed = np.asarray(consensus['canonical_signs'])[keep]
    targets = best_equivalent_forms(anchor, reference)
    counts = np.count_nonzero(targets[:, keep] == observed, axis=1)
    return {'retained_positions': len(keep), 'reference_matches_by_alignment': counts.tolist()}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))

    first, second = [artifact('v106-leo-profiled-word-' + suffix, 'E108', 'leo_profiled_word.json')
                     for suffix in ('final', 'confirm')]
    if first != second:
        raise ValueError('Blind replicas differ')
    expected = artifact('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    reference = np.asarray(expected['cycle_folds'][0]['signed_template'])
    pairs = []
    for pair in first['pairs']:
        baseline = consensus_comparison(pair['arms'][0]['initial_word'], pair['initial_consensus'], reference)
        refined = consensus_comparison(pair['arms'][0]['refined_word'], pair['refined_consensus'], reference)
        arms = []
        for arm in pair['arms']:
            scores = np.asarray(arm['scores']['donor'])
            controls = np.count_nonzero(scores[:, 2:] >= scores[:, 1, None], axis=1)
            arms.append({'guard': arm['guard'], 'learning_frames': arm['learning_frames'],
                         'initial_reference': cyclic_distance(np.asarray(arm['initial_word']), reference),
                         'refined_reference': cyclic_distance(np.asarray(arm['refined_word']), reference),
                         'donor_gain_initial_and_refined': scores[:, :2].tolist(),
                         'donor_controls_at_or_above_refined': controls.tolist()})
        pairs.append({'time_s': pair['time_s'], 'initial_consensus': baseline,
                      'refined_consensus': refined, 'arms': arms})
        print(pair['time_s'], 'retained/matches', baseline, '->', refined)
    result = {'schema': 'starlink-leo-profiled-word-comparison/v1', 'pairs': pairs,
              'inputs_sha256': {p.as_posix(): h for p, h in inputs.items()}}
    out = experiment_output('E108')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'leo_profiled_word_comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
