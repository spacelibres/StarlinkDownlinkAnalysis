"""Compare frozen negative-side predictions with prior triad decisions."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_slot_frame_power import CONTROL_SHIFTS, DEGREES
from audit_wav_word_state import build_symbols, relative_symbols


def label(row):
    if np.any(row == 0):
        return 'erased'
    return ''.join('+' if value > 0 else '-' for value in row)


def compare_patterns(observed, predicted):
    complete = np.all(observed != 0, axis=1)
    known = observed[complete]
    trials = predicted[:, complete]
    consensus = np.where(np.all(trials == trials[:1], axis=0), trials[0], 0)
    matrices = []
    for trial in (*trials, consensus):
        counts = defaultdict(Counter)
        for truth, hypothesis in zip(known, trial, strict=True):
            counts[label(truth)][label(hypothesis)] += 1
        matrices.append({key: dict(sorted(values.items())) for key, values in sorted(counts.items())})
    return matrices[:-1], matrices[-1]


def main():
    require_run_context()
    inputs = {}

    def artifact(run, experiment, filename):
        return read(checked_artifact(run, experiment, filename, inputs))

    result = artifact('v114-wav-word-state', 'E116', 'wav_word_state.json')
    confirm = artifact('v114-wav-word-state-confirm', 'E116', 'wav_word_state.json')
    if result != confirm:
        raise ValueError('Frozen E116 wave fits disagree')
    prior = artifact('v112-wav-slot-frame-power-final', 'E114', 'wav_slot_frame_power.json')
    if prior != artifact('v112-wav-slot-frame-power-confirm', 'E114', 'wav_slot_frame_power.json'):
        raise ValueError('Prior E114 frame fits disagree')
    geometry = artifact('v040-wav-tessellation', 'E042', 'wav_tessellation.json')
    tail = artifact('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    resources = np.asarray(geometry['reference_comparison']['resource_index_by_lane_and_slot_class'])
    carrier = geometry['carrier_indices'].index(result['carrier'])
    fixed = result['fixed_phase_subtracted']
    slots = result['slot_classes']
    words = build_symbols(np.asarray(tail['cycle_folds'][0]['signed_template']), resources, carrier, slots)
    relative = relative_symbols(words, fixed, carrier, slots)
    observed = []
    phases = []
    control_groups = [[] for _ in CONTROL_SHIFTS]
    for fitted, before in zip(result['windows'], prior['windows'], strict=True):
        if fitted['ordinals'] != before['ordinals_in_window'] or fitted['window_start_s'] != before['window_start_s']:
            raise ValueError('Prior decisions and new predictions refer to different frames')
        observed.append(np.column_stack([before['retained_signs'][str(slot)] for slot in slots]))
        phase = np.array([fitted['fits'][degree][arm]['models']['negative_side_shift']['phase']
                          for degree in DEGREES for arm in (0, 1)])
        phases.append(phase)
        for part, lag in zip(control_groups, CONTROL_SHIFTS, strict=True):
            part.append(np.roll(phase, lag, axis=1))
    observed = np.concatenate(observed)
    phases = np.concatenate(phases, axis=1)
    result_by_arm, consensus = compare_patterns(observed, relative[phases])
    controls = [compare_patterns(observed, relative[np.concatenate(part, axis=1)])[1] for part in control_groups]
    complete = np.all(observed != 0, axis=1)
    outcomes_by_observed_label = {}
    for truth in ('+++', '---'):
        matched = np.all(observed == (1 if truth == '+++' else -1), axis=1)
        outcomes_by_observed_label[truth] = [
            dict(sorted(Counter(a[matched].tolist()).items())) for a in phases]
    output = {'schema': 'starlink-wav-word-triad-compare/v1',
              'inputs_sha256': {path.as_posix(): value for path, value in inputs.items()},
              'complete_observed_count': int(complete.sum()), 'by_arm': result_by_arm, 'consensus': consensus,
              'control_frame_rolls': list(CONTROL_SHIFTS), 'control_consensus': controls,
              'negative_phase_histogram_by_observed_label': outcomes_by_observed_label,
              'nominal_dominant_phase_signs': {key: label(relative[key]) for key in (0, fixed, 16)}}
    destination = experiment_output('E116')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'wav_word_triad_compare.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n',
                                                           encoding='utf-8')
    print(consensus, flush=True)


if __name__ == '__main__':
    main()
