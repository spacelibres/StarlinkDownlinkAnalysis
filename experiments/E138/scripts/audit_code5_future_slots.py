"""Align Code 5 tail nibbles with subsequent verified frame counters."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import raw_record_set
from audit_code5_record_countdown import TAIL_BYTES
from audit_leo_layout_relation import archive_words
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


SEED = 20261121138
FRAME_RATE_HZ = 750
TOA_TOLERANCE_FRAMES = .05
GRID_STEP = 4
# Each control changes one part of the frozen mapping.
ARMS = {'low_first_4': (False, 4, 0), 'high_first_4': (True, 4, 0),
        'stride_3': (False, 3, 0), 'stride_5': (False, 5, 0), 'stride_8': (False, 8, 0),
        'one_frame_early': (False, 4, -1), 'one_frame_late': (False, 4, 1)}


def decode_nibbles(data, observed, high_first=False):
    codes = []
    if len(observed) != len(data) * 8:
        raise ValueError('One observation flag per bit is required')
    order = (4, 0) if high_first else (0, 4)
    for i, byte in enumerate(data):
        for shift in order:
            mask = observed[8 * i + shift:8 * i + shift + 4]
            codes.append((byte >> shift) & 15 if all(mask) else None)
    return codes


def cycle_end(counter, period, phase):
    return counter + (phase - counter - 1) % period + 1


def predict_slots(counter, deadline, first_observed, last_observed, codes, step, shift):
    predictions = []
    for i, code in enumerate(codes[1:], 1):
        if counter + GRID_STEP * i >= deadline:
            break
        target = counter + step * i + shift
        if code is not None and counter < target < deadline and first_observed <= target <= last_observed:
            predictions.append((i, target, code))
    return predictions


def source_masks(rows, words):
    if sorted(words) != [r['frame'] for r in rows]:
        raise ValueError('E137 capture and Code 5 word memberships do not match')
    for r in rows:
        pair = words[r['frame']]
        data = pair[0][0] + pair[1][0]
        flags = np.concatenate((pair[0][1], pair[1][1]))
        start = r['length'] + r['span'] - TAIL_BYTES + 36
        if data[start:start + 8].hex() != r['tail_slots']:
            raise ValueError(f'E137 tail bytes differ in frame {r["frame"]}')
        r['slot_bits_observed'] = flags[8 * start:8 * (start + 7)].tolist()


def add_targets(rows, timing):
    for row in rows:
        row.update(timing[row['frame']])


def indexed_targets(rows):
    indexed = defaultdict(list)
    for row in rows:
        if row['counter_observed'] and row['toa_s'] is not None:
            indexed[row['counter']].append(row)
    return indexed


def trials(source_rows, targets, period, phase, arm):
    high_first, step, shift = arm
    first, last = min(targets), max(targets)
    cases = []
    for row in source_rows:
        if not row['word'] or not row['word_observed'] or not row['counter_observed'] or row['toa_s'] is None:
            continue
        tail = bytes.fromhex(row['tail_slots'])
        values = decode_nibbles(tail[:7], row['slot_bits_observed'], high_first)
        n = row['counter']
        end = cycle_end(n, period, phase)
        for i, t, code in predict_slots(n, end, first, last, values, step, shift):
            aligned, off_axis = [], []
            for target in targets.get(t, ()):
                error = (target['toa_s'] - row['toa_s']) * FRAME_RATE_HZ - (t - n)
                matched = {'frame': target['frame'], 'toa_residual_frames': error,
                           'span': target['span'], 'code5_allocation_count': target['code5_allocation_count']}
                (aligned if abs(error) <= TOA_TOLERANCE_FRAMES else off_axis).append(matched)
            cases.append({'cycle_end': end, 'source_frame': row['frame'], 'source_counter': n,
                          'index': i, 'target_counter': t, 'code': code, 'aligned': aligned, 'off_axis': off_axis})
    return cases


def summarize(cases, common=None):
    values = defaultdict(Counter)
    target_codes = defaultdict(set)
    target_hits = defaultdict(set)
    layout_by_code = defaultdict(Counter)
    paths = []
    for r in cases:
        if common is not None and (r['source_frame'], r['index']) not in common:
            continue
        code = r['code']
        key = (r['cycle_end'], r['target_counter'])
        target_codes[key].add(code)
        target_hits[key].update(t['frame'] for t in r['aligned'])
        values[code]['proposals'] += 1
        values[code]['proposals_with_observation'] += bool(r['aligned'])
        values[code]['proposals_with_off_axis_only'] += bool(r['off_axis']) and not r['aligned']
        for hit in r['aligned']:
            paths.append(abs(hit['toa_residual_frames']))
            layout_by_code[code][(hit['span'], hit['code5_allocation_count'])] += 1
    statuses = Counter(tuple(sorted(codes)) for codes in target_codes.values())
    counts = {}
    for code, value in sorted(values.items()):
        pure = [key for key, codes in target_codes.items() if codes == {code}]
        counts[str(code)] = {**value, 'unanimous_targets': len(pure),
                             'observed_unanimous_targets': sum(bool(target_hits[key]) for key in pure),
                             'observed_span_and_allocation_counts': [
                                 {'span': span, 'code5_allocation_count': count, 'occurrences': frequency}
                                 for (span, count), frequency in sorted(layout_by_code[code].items())]}
    return {'code_counts': counts, 'unique_targets': len(target_codes),
            'conflicting_targets': sum(count for codes, count in statuses.items() if len(codes) > 1),
            'max_abs_toa_residual_frames': max(paths) if paths else None}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    preceding = read(artifact('v135-code5-record-countdown', 'E137', 'code5_record_countdown.json'))
    directory = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    qin = preceding['captures']['Qin']
    model = read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    legacy = 'v010-code5-parity-02'
    ut = 'v022-ut-replay'
    with np.load(artifact(legacy, 'E012', 'code5_observations.npz'), allow_pickle=False) as code:
        generator = code['parity_generator']
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    source_masks(qin, archive_words(artifact, pn, generator, directory))
    qin_headers = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    timing = {}
    for row in qin_headers:
        h1 = row['h1']
        first = h1['layout20']['entries'][0]
        if not h1['crc_ok'] or not h1['layout20']['packing_valid'] or first['mcs_candidate'] != 5:
            raise ValueError(f'Qin allocation does not match the frozen CRC-checked frame {row["frame"]}')
        timing[row['frame']] = {'toa_s': row['toa_s'], 'code5_allocation_count': first['quantity']}
    add_targets(qin, timing)
    period = preceding['qin_model']['period']
    phase = preceding['qin_model']['phase']
    ends = sorted({cycle_end(r['counter'], period, phase) for r in qin})[1:-1]
    rng = np.random.default_rng(SEED)
    held = sorted(int(i) for i in rng.permutation(ends)[:len(ends) // 3])
    comparison = sorted(set(ends) - set(held))

    panels, receipts = {}, {}
    for label, partition in (('comparison', comparison), ('holdout', held)):
        selected = [r for r in qin if cycle_end(r['counter'], period, phase) in partition]
        archive_index = indexed_targets(selected)
        arms = {name: trials(selected, archive_index, period, phase, spec) for name, spec in ARMS.items()}
        common = set.intersection(*({(r['source_frame'], r['index']) for r in cases} for cases in arms.values()))
        missing_masks = {r['frame']: [i for i, code in enumerate(decode_nibbles(bytes.fromhex(r['tail_slots'])[:7],
                         r['slot_bits_observed'])) if code is None] for r in selected if not all(r['slot_bits_observed'])}
        panels[label] = {'cycles': partition, 'frames': [r['frame'] for r in selected], 'common_opportunities': len(common),
                         'erased_nibbles_by_frame': missing_masks,
                         'arms': {name: {'all': summarize(cases), 'common': summarize(cases, common)}
                                  for name, cases in arms.items()}}
        receipts[label] = arms['low_first_4']

    raw = {}
    for label, run, exp, npz, report, key in (
            ('480MSps', legacy, 'E012', 'code5_observations.npz', 'code5_parity.json', 'hard_bits'),
            ('250MSps', ut, 'E014', 'ut_observations.npz', 'ut_transfer.json', 'hard_code5_words')):
        meta = read(artifact(run, exp, report))
        words = raw_record_set(artifact(run, exp, npz), meta['words'], key, pn, generator)
        rows = preceding['captures'][label]
        source_masks(rows, words)
        if label == '480MSps':
            anchors = read(artifact('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']
            clock = {r['legacy_frame_index']: {'toa_s': r['pss_start_sample'] / 480e6 if r['pss_start_observed'] else None,
                         'code5_allocation_count': r['first_layout20']['entries'][0]['quantity']} for r in anchors}
        else:
            clock = {r['candidate_index']: {'toa_s': r['pss']['epoch'] / meta['resampling']['rate_hz'],
                         'code5_allocation_count': r['h1']['layout20']['entries'][0]['quantity']}
                     for r in meta['candidates'] if r.get('h1', {}).get('crc_ok') and r['pss']['epoch'] is not None}
        add_targets(rows, clock)
        local_phase = preceding['raw_tests'][label]['phase']
        target = indexed_targets(rows)
        cases = trials(rows, target, period, local_phase, ARMS['low_first_4'])
        raw[label] = {'summary': summarize(cases), 'proposals': cases}

    result = {'schema': 'starlink-code5-future-slots/v1', 'seed': SEED, 'period': period, 'phase': phase,
              'frame_rate_hz': FRAME_RATE_HZ, 'toa_tolerance_frames': TOA_TOLERANCE_FRAMES,
              'input_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              'qin': panels, 'qin_low_first_4_proposals': receipts, 'raw': raw}
    print('Qin holdout cycles', len(held), 'common comparisons', panels['holdout']['common_opportunities'])
    for name, arm in panels['holdout']['arms'].items():
        report = arm['all']
        print(name, {i: (r['proposals_with_observation'], r['proposals']) for i, r in report['code_counts'].items()},
              'conflicting targets', report['conflicting_targets'])
    print('Raw', {name: value['summary'] for name, value in raw.items()})
    out = experiment_output('E138')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'code5_future_slots.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
