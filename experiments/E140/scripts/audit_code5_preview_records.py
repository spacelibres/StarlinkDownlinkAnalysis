"""Link the future-slot bitmap to an exact conversion between Code 5 records."""

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_future_slots import FRAME_RATE_HZ, TOA_TOLERANCE_FRAMES, decode_nibbles
from audit_code5_later_records import interior_records, load_evidence
from audit_code5_record_countdown import TAIL_BYTES
from audit_code5_slot_bitmap import bitmap_offsets, build_index, same_axis
from audit_ut_repeated_data import checked_artifact, read


def lane_positions(length, start, period, width):
    return list(range(start, 8 * length - width + 1, period))


def transform_preview(payload, start, period, width, preserve=2):
    if payload[0] != 3:
        raise ValueError('Expected a type-03 preview record')
    value = int.from_bytes(payload, 'little') ^ 2
    positions = lane_positions(len(payload), start, period, width)
    limit = max(0, len(positions) - preserve)
    for bit in positions[:limit]:
        value &= ~(((1 << width) - 1) << bit)
    return value.to_bytes(len(payload), 'little')


def observed_mask(code, record):
    start = record.start
    end = start + len(record.payload)
    masks = [code[i][1] for i in range(start // 128, (end - 1) // 128 + 1)]
    aligned_start = (start % 128) * 8
    flags = np.concatenate(masks)[aligned_start:aligned_start + len(record.payload) * 8]
    if int(flags.sum()) != record.observed_bits:
        raise ValueError('Record observation mask does not match its receipt')
    return flags


def bit_differences(a, b):
    left = np.frombuffer(a, dtype=np.uint8)
    right = np.frombuffer(b, dtype=np.uint8)
    return np.unpackbits(left ^ right, bitorder='little').astype(bool)


def learn_lanes(pairs):
    changed = set()
    for source, target, observed in pairs:
        differences = np.flatnonzero(bit_differences(source.payload, target.payload) & observed)
        changed.update(int(i) for i in differences if i >= 8)
    start = min(changed)
    models = []
    for period in range(24, 97):
        residues = sorted({(i - start) % period for i in changed})
        models.append({'period': period, 'residues': residues})
    models.sort(key=lambda m: Fraction(len(m['residues']), m['period']))
    best, runner_up = models[:2]
    if (Fraction(len(best['residues']), best['period']) >= Fraction(len(runner_up['residues']), runner_up['period']) or
            best['residues'] != list(range(len(best['residues'])))):
        raise ValueError('Training does not support unique contiguous difference lanes')
    return start, best['period'], len(best['residues']), models


def summarize_pairs(rows):
    return {'pairs': len(rows), 'full_record_matches': sum(r['exact'] for r in rows),
            'fully_observed_both_records': sum(r['joint_direct_bits'] == r['length'] * 8 for r in rows),
            'joint_direct_bits': sum(r['joint_direct_bits'] for r in rows),
            'direct_prediction_errors': sum(r['direct_errors'] for r in rows),
            'candidate_bits': sum(r['length'] * 8 for r in rows)}


def terminal_shapes(records, start, period, width):
    counts = defaultdict(Counter)
    for record in records:
        if record.number < 3 or record.payload[0] not in (1, 3):
            continue
        kind = f'{record.payload[0]:02x}'
        value = int.from_bytes(record.payload, 'little')
        positions = lane_positions(len(record.payload), start, period, width)
        lanes = [(value >> bit) & ((1 << width) - 1) for bit in positions]
        counts[kind]['records'] += 1
        if lanes:
            counts[kind]['last_lane_is_256'] += lanes[-1] == 256
        if len(lanes) >= 2:
            counts[kind]['penultimate_lane_is_256'] += lanes[-2] == 256
        if len(lanes) >= 3:
            counts[kind]['all_lanes_before_last_two_zero'] += not any(lanes[:-2])
        counts[kind]['no_complete_lane'] += not lanes
    return {k: dict(v) for k, v in counts.items()}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    def load(run, exp, name):
        return read(artifact(run, exp, name))

    a = load('v099-code5-later-records', 'E101', 'code5_later_records.json')
    directory = load('v098-code5-records-final', 'E100', 'code5_directory_records.json')
    clocks = load('v095-code5-clock-final', 'E097', 'code5_clock_correction.json')
    timing = load('v135-code5-record-countdown', 'E137', 'code5_record_countdown.json')
    schedule = load('v136-code5-future-slots-final', 'E138', 'code5_future_slots.json')
    bitmap = load('v137-code5-slot-bitmap', 'E139', 'code5_slot_bitmap.json')
    if set(bitmap['archived_frames']['holdout']) != set(schedule['qin']['holdout']['frames']):
        raise ValueError('Schedule partitions do not match the slots experiment')
    frames, pages = load_evidence(artifact)
    if pages != directory['archive_source']['pages'] or pages != a['archive_source']['pages']:
        raise ValueError('Archive IQ-derived pages changed')
    archive = frames['archive']
    records, missing = interior_records(archive, directory['archive'], clocks['archive'])
    if [list(row) for row in missing] != a['unavailable_records']['archive'] or len(records) != len(a['records']['archive']):
        raise ValueError('Record inventory differs from E101')
    by_key = {(r.frame, r.number): r for r in records}
    known_toa = {r['frame']: r['toa_s'] for r in load('v034-qin-header-survey', 'E036', 'qin_header_survey.json')['frames']}
    qin = {r['frame']: {**r, 'toa_s': known_toa[r['frame']]} for r in timing['captures']['Qin']}
    index = build_index(qin.values())
    partitions = {name: set(row['frames']) for name, row in schedule['qin'].items()}
    source_receipts = {tuple(r['source']): r for r in a['qin_future_pairs']['0']}
    zero = []
    for row in directory['archive']:
        frame = row['frame']
        if qin[frame]['word'] == 0 or len(row['lengths']) == 2:
            zero.append({'frame': frame, 'timing_word': qin[frame]['word'], 'directory_bytes': row['directory_length'],
                         'allocations': row['word_count_h1'], 'record_bytes': row['lengths'],
                         'padding_bytes': row['padding_bytes'], 'padding_zero': row['padding_zero'],
                         'complete': row['complete_allocation']})
    erased = {int(frame): indices for p in schedule['qin'].values() for frame, indices in p['erased_nibbles_by_frame'].items()}
    preview_slots = []
    for row in a['records']['archive']:
        if row['class'] != 'type_03' or not row['five_header_bytes_observed']:
            continue
        frame = row['frame']
        curr = qin[frame]
        counter = row['record_counter']
        delta = counter - curr['counter']
        if delta != row['delta'] or not curr['counter_observed'] or not curr['word_observed']:
            raise ValueError('Observed preview record counter differs from source clocks')
        slot, offset = divmod(delta, 4)
        if not 1 <= delta <= 4 or not curr['word'] or slot in erased.get(frame, ()):
            raise ValueError('Preview counter is outside the checked family')
        byte = curr['length'] + curr['span'] - TAIL_BYTES + 36 + slot // 2
        word_index, byte_in_word = divmod(byte, 128)
        bit_in_word = 8 * byte_in_word + 4 * (slot % 2)
        observed_nibble = bool(archive[frame][word_index][1][bit_in_word:bit_in_word + 4].all())
        values = decode_nibbles(bytes.fromhex(curr['tail_slots'])[:7], [True] * 56)
        target = same_axis(curr, counter, index)
        receipt = source_receipts.get((frame, row['number']))
        preview_slots.append({'source': [frame, row['number']], 'delta': delta, 'counter': counter,
                              'code': values[slot] if observed_nibble else None,
                              'bit_set': offset in bitmap_offsets(values[slot]) if observed_nibble else None,
                              'aligned_frames': target['aligned'], 'other_axis_frames': target['off_axis'],
                              'E101_target': receipt['target'] if receipt else None})

    pairs = []
    for key, row in source_receipts.items():
        source = by_key[key]
        target = by_key[tuple(row['target'])]
        if len(source.payload) != len(target.payload) or source.payload[0] != 3 or target.payload[0] != 1:
            raise ValueError('Original pair has a different layout or type')
        delta = qin[target.frame]['counter'] - qin[source.frame]['counter']
        error = (known_toa[target.frame] - known_toa[source.frame]) * FRAME_RATE_HZ - delta
        group = next((name for name, members in partitions.items() if source.frame in members and target.frame in members), 'other')
        pairs.append((source, target, error, group,
                      observed_mask(archive[source.frame], source) & observed_mask(archive[target.frame], target)))
    learning = [(x, y, direct) for x, y, e, group, direct in pairs if abs(e) <= TOA_TOLERANCE_FRAMES and group == 'comparison']
    start, period, width, models = learn_lanes(learning)
    shapes = {'Qin': terminal_shapes(records, start, period, width)}
    for name in ('250MSps', '480MSps'):
        raw, _ = interior_records(frames[name], directory['raw'][name], clocks['raw'][name])
        shapes[name] = terminal_shapes(raw, start, period, width)
    results = []
    controls = Counter()
    summary = defaultdict(list)
    for source, target, err, split, direct in pairs:
        qualified = abs(err) <= TOA_TOLERANCE_FRAMES
        predicted = transform_preview(source.payload, start, period, width)
        mismatch = bit_differences(predicted, target.payload)
        lane = lane_positions(len(source.payload), start, period, width)
        value = int.from_bytes(source.payload, 'little')
        row = {'source': [source.frame, source.number], 'target': [target.frame, target.number], 'length': len(predicted),
               'toa_error_frames': err, 'same_axis': qualified, 'partition': split, 'exact': not bool(mismatch.any()),
               'predicted_sha256': sha256(predicted).hexdigest(), 'target_sha256': sha256(target.payload).hexdigest(),
               'direct_errors': int((mismatch & direct).sum()), 'joint_direct_bits': int(direct.sum()),
               'source_terminal_values': [(value >> bit) & ((1 << width) - 1) for bit in lane[-2:]],
               'cleared_lanes': len(lane[:-2]), 'mismatch_positions': np.flatnonzero(mismatch).tolist()}
        results.append(row)
        summary[split if qualified else 'different_axis'].append(row)
        if qualified and split == 'holdout':
            variants = {'type_only': bytes([1]) + source.payload[1:],
                        'stride_minus_one': transform_preview(source.payload, start, period-1, width),
                        'stride_plus_one': transform_preview(source.payload, start, period+1, width),
                        'clear_terminal_lanes_too': transform_preview(source.payload, start, period, width, preserve=0)}
            controls.update({name: int(expected == target.payload) for name, expected in variants.items()})

    for shift, links in a['qin_future_pairs'].items():
        if shift == '0':
            continue
        controls[f'counter_shift_{shift}_pairs'] = len(links)
        controls[f'counter_shift_{shift}_exact'] = sum(transform_preview(by_key[tuple(r['source'])].payload,
            start, period, width) == by_key[tuple(r['target'])].payload for r in links)
    slot_summary = defaultdict(Counter)
    for row in preview_slots:
        frame = row['source'][0]
        group = next((name for name, members in partitions.items() if frame in members), 'other')
        state = {True: 'set', False: 'clear', None: 'erased'}[row['bit_set']]
        key = f'delay{row["delta"]}_{state}'
        for scope in ('all', group):
            slot_summary[scope][key + '_count'] += 1
            slot_summary[scope][key + '_aligned'] += bool(row['aligned_frames'])
            slot_summary[scope][key + '_off_axis_only'] += not row['aligned_frames'] and bool(row['other_axis_frames'])
    report = {'schema': 'starlink-code5-preview-records/v1',
              'inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'lane_model': {'start': start, 'period': period, 'width': width, 'period_candidates': models,
                             'preserve_last_complete_lanes': 2},
              'archive_page_receipts': pages, 'zero_and_two_record_cases': zero, 'terminal_shapes': shapes,
              'preview_slots': preview_slots, 'slot_summary': {k: dict(v) for k, v in slot_summary.items()},
              'summary_by_pair_partition': {k: summarize_pairs(v) for k, v in summary.items()},
              'pair_predictions': results, 'exact_match_controls': dict(controls)}
    print('Learned lanes:', start, period, width, 'from', len(learning), 'comparison pairs', flush=True)
    print(report['summary_by_pair_partition'], flush=True)
    print('Slots:', report['slot_summary']['all'], flush=True)
    print('Controls:', report['exact_match_controls'], flush=True)
    out = experiment_output('E140')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'code5_preview_records.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
