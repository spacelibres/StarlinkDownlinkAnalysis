"""Compare archived blind words with the wideband reference after estimation."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_leo_blind_word import best_equivalent_forms, cyclic_distance
from audit_ut_repeated_data import checked_artifact, read


def main():
    require_run_context()
    inputs = {}
    measured = [read(checked_artifact(run, 'E066', 'leo_blind_word.json', inputs))
                for run in ('v064-leo-blind-consensus', 'v064-leo-blind-consensus-confirm')]
    if measured[0] != measured[1]:
        raise ValueError('Blind-run receipts disagree')
    broadband = read(checked_artifact('v023-binary-tail-final', 'E025', 'ut_binary_tail.json', inputs))
    reference = np.asarray(broadband['cycle_folds'][0]['signed_template'])
    pairs = []
    for pair in measured[0]['pairs']:
        words = [np.array(arm['learned_word']) for arm in pair['modes']]
        comparisons = []
        for word, arm in zip(words, pair['modes'], strict=True):
            matched = cyclic_distance(word, reference)
            controls = [cyclic_distance(np.asarray(control), reference)['mismatches']
                        for control in arm['control_words']]
            comparisons.append({'guard': arm['guard'], 'learned_on_frames': arm['learning_frames'],
                                **matched, 'control_distances': controls,
                                'controls_no_farther': sum(distance <= matched['mismatches'] for distance in controls)})
        distances = [[cyclic_distance(a, b)['mismatches'] for b in words] for a in words]
        consensus = pair['blind_consensus']
        keep = consensus['retained_positions']
        sign = np.asarray(consensus['canonical_signs'])[keep]
        aligned = best_equivalent_forms(words[0], reference)
        counts = np.count_nonzero(aligned[:, keep] != sign, axis=1)
        pairs.append({'time_s': pair['time_s'], 'wideband_comparison_after_estimation': comparisons,
                      'blind_fold_pair_distances': distances, 'blind_consensus_positions': keep,
                      'consensus_mismatches_at_full_anchor_best_alignments': counts.tolist()})
        print(pair['time_s'], [r['mismatches'] for r in comparisons], 'inter-arm distances', distances, flush=True)
    out = experiment_output('E066')
    out.mkdir(parents=True, exist_ok=True)
    (out/'leo_blind_word_reference_comparison.json').write_text(json.dumps(
        {'schema': 'leo-blind-word-reference-comparison/v1',
         'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}, 'pairs': pairs},
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
