"""Test the four bit positions of each Code 5 future-slot nibble."""

from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_future_slots import FRAME_RATE_HZ, TOA_TOLERANCE_FRAMES, cycle_end, decode_nibbles
from audit_ut_repeated_data import checked_artifact, read


DISCOVERY_CYCLE_END = 5830
RARE_CODES = (2, 4, 8)


def bitmap_offsets(mask):
    return tuple(i for i in range(4) if mask & (1 << i))


def classify_quartet(base, deadline):
    if base >= deadline:
        return 'beyond'
    return 'within' if base + 3 < deadline else 'crossing'


def same_axis(row, counter, index):
    aligned, off_axis = [], []
    for target in index.get(counter, ()):
        error = (target['toa_s'] - row['toa_s']) * FRAME_RATE_HZ - (counter - row['counter'])
        item = {'frame': target['frame'], 'residual_frames': error}
        (aligned if abs(error) <= TOA_TOLERANCE_FRAMES else off_axis).append(item)
    return {'counter': counter, 'aligned': aligned, 'off_axis': off_axis}


def build_index(rows):
    index = defaultdict(list)
    for row in rows:
        if row['counter_observed'] and row['toa_s'] is not None:
            index[row['counter']].append(row)
    return index


def build_quartets(rows, index, period, phase, erased):
    first, last = min(index), max(index)
    results = []
    for row in rows:
        if not row['counter_observed'] or not row['word_observed'] or row['toa_s'] is None:
            continue
        n = row['counter']
        end = cycle_end(n, period, phase)
        codes = decode_nibbles(bytes.fromhex(row['tail_slots'])[:7], [True] * 56)
        for i in erased.get(row['frame'], ()):
            codes[i] = None
        for i, code in enumerate(codes[1:], 1):
            base = n + 4 * i
            if code is None or base < first or base + 3 > last:
                continue
            results.append({'frame': row['frame'], 'counter': n, 'timing_word_zero': row['word'] == 0,
                            'cycle_end': end, 'index': i, 'code': code, 'base_counter': base,
                            'region': classify_quartet(base, end),
                            'trials': [same_axis(row, base + d, index) for d in range(4)]})
    return results


def summarize(rows):
    trials = Counter()
    hits = Counter()
    unique = defaultdict(set)
    for row in rows:
        for d, target in enumerate(row['trials']):
            code = row['code']
            trials[(code, d)] += 1
            hits[(code, d)] += bool(target['aligned'])
            if target['aligned']:
                unique[(code, d)].add(target['counter'])
    return {str(code): [{'offset': d, 'proposals': trials[(code, d)], 'matches': hits[(code, d)],
                         'distinct_matched_counters': len(unique[(code, d)])} for d in range(4)]
            for code in sorted({r['code'] for r in rows})}


def unique_associations(rows):
    mapped = defaultdict(set)
    observed = {}
    for row in rows:
        for d, target in enumerate(row['trials']):
            key = (row['cycle_end'], target['counter'])
            mapped[key].add(d in bitmap_offsets(row['code']))
            observed[key] = observed.get(key, False) or bool(target['aligned'])
    return {'total_targets': len(mapped), 'conflicting_targets': sum(len(s) > 1 for s in mapped.values()),
            'unanimously_set': sum(s == {True} for s in mapped.values()),
            'unanimously_clear': sum(s == {False} for s in mapped.values()),
            'observed_unanimously_set': sum(observed[k] for k, flags in mapped.items() if flags == {True}),
            'observed_unanimously_clear': sum(observed[k] for k, flags in mapped.items() if flags == {False})}


def test_permutations(rows):
    return [{'offset_by_bit': mapping, 'matches': sum(bool(r['trials'][mapping[bitmap_offsets(r['code'])[0]]]['aligned'])
             for r in rows)} for mapping in permutations(range(4))]


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    timing = read(artifact('v135-code5-record-countdown', 'E137', 'code5_record_countdown.json'))
    previous = read(artifact('v136-code5-future-slots-final', 'E138', 'code5_future_slots.json'))
    archive = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    by_frame = {r['frame']: r for r in archive}
    qin = timing['captures']['Qin']
    for row in qin:
        observed = by_frame[row['frame']]
        if not observed['h1']['crc_ok']:
            raise ValueError('Missing CRC record for archive target')
        row['toa_s'] = observed['toa_s']
    partition = {name: set(p['frames']) for name, p in previous['qin'].items()}
    if partition['holdout'] & partition['comparison']:
        raise ValueError('Original whole-cycle partition changed')
    frame_ids = partition['holdout'] | partition['comparison']
    erased = {int(frame): indices for p in previous['qin'].values() for frame, indices in p['erased_nibbles_by_frame'].items()}
    rows = [r for r in qin if r['frame'] in frame_ids]
    index = build_index(qin)
    period, phase = previous['period'], previous['phase']
    quartets = build_quartets(rows, index, period, phase, erased)
    active = [r for r in quartets if not r['timing_word_zero']]
    within = [r for r in active if r['region'] == 'within']
    rare = [r for r in within if r['code'] in RARE_CODES]
    outside_four = [r for r in active if r['region'] == 'beyond' and r['code'] == 4]
    outside_eight = [r for r in active if r['region'] == 'beyond' and r['code'] == 8]
    prior = {r['frame']: r for r in qin}
    for label, samples in previous['qin_low_first_4_proposals'].items():
        used = {(r['source_frame'], r['index']): (r['target_counter'], r['code']) for r in samples}
        if any(used.get((r['frame'], r['index'])) != (r['base_counter'], r['code'])
               for r in within if r['frame'] in partition[label]):
            raise ValueError('Whole-quartet subset differs from E138 source mapping')
    if {r['cycle_end'] for r in rare if r['frame'] == 42} != {DISCOVERY_CYCLE_END}:
        raise ValueError('Missing pilot group that suggested the bit-order hypothesis')

    results = {'within_active': summarize(within),
               'within_timing_word_zero': summarize([r for r in quartets if r['region'] == 'within' and r['timing_word_zero']]),
               'beyond_active': summarize([r for r in active if r['region'] == 'beyond']),
               'outside_discovery_rare': summarize([r for r in rare if r['cycle_end'] != DISCOVERY_CYCLE_END]),
               'all_rare': summarize(rare),
               'E138_held_rare': summarize([r for r in rare if r['frame'] in partition['holdout']]),
               'full_within_active': unique_associations(within),
               'rare_offset_permutations': test_permutations([r for r in rare if r['cycle_end'] != DISCOVERY_CYCLE_END]),
               'code_4_cross_boundary': {'proposals': len(outside_four),
                   'same_next_cycle_phase4': sum(r['base_counter'] + 2 == r['cycle_end'] + 4 for r in outside_four)},
               'timing_word_zero_cycle_phase_mod4': dict(Counter((r['counter'] % period) % 4
                    for r in rows if not r['word']))}

    legacy = read(artifact('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']
    ps = {r['legacy_frame_index']: r['pss_start_sample'] / 480e6 if r['pss_start_observed'] else None for r in legacy}
    old = timing['captures']['480MSps']
    for r in old:
        r['toa_s'] = ps[r['frame']]
    raw_index = build_index(old)
    raw_proposals = previous['raw']['480MSps']['proposals']
    # Only inspect directly observed code-1 entries retained by the earlier IQ replay.
    raw_by_frame = {r['frame']: r for r in old}
    raw_four = [{'frame': r['source_frame'], 'code': 1,
                 'trials': [same_axis(raw_by_frame[r['source_frame']], r['target_counter'] + d, raw_index) for d in range(4)]}
                for r in raw_proposals if r['code'] == 1 and r['target_counter'] + 3 <= max(raw_index)]
    results['480MSps_within_recorded_span'] = summarize(raw_four)
    result = {'schema': 'starlink-code5-slot-bitmap/v1', 'period': period, 'discovery_cycle_end': DISCOVERY_CYCLE_END,
              'input_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'archived_frames': {name: sorted(frames) for name, frames in partition.items()},
              'summary': results, 'rare_quartets': rare, 'cross_boundary_four': outside_four,
              'cross_boundary_eight': outside_eight, 'raw_positive_quartets': raw_four,
              'qin_quartets_index': [[r['frame'], r['index'], r['code'], r['region'],
                  [bool(t['aligned']) for t in r['trials']], prior[r['frame']]['word']] for r in quartets]}
    print('Within-cycle rare:', results['all_rare'])
    print('Outside discovery:', results['outside_discovery_rare'])
    print('Original periods held out:', results['E138_held_rare'])
    print('Cross boundary code 4:', results['code_4_cross_boundary'], results['beyond_active']['4'])
    print('480 MS/s adjacent frame detection:', results['480MSps_within_recorded_span'])
    out = experiment_output('E139')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'code5_slot_bitmap.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
