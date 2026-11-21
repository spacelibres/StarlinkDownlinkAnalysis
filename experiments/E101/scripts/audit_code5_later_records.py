"""Audit later Code 5 record formats against independent frame counters."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, difference_gcds
from audit_code5_directory_records import quantity, raw_words, recover_archive, span
from audit_code5_frame_count import descramble
from audit_h2_projection import visible_positions
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


ORIENTATIONS = ('lsb_first', 'msb_first', 'reversed_trailer_bytes')
MIN_DISTINCT_CRC = 8
COUNTER_SHIFT = 11
COUNTER_RECORD_BIT = 8 + COUNTER_SHIFT
COUNTER_WIDTH = 14
PAIR_PREFIX_END = 20
PAIR_SHIFT = (-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8)


@dataclass(frozen=True)
class Record:
    frame: int
    number: int
    start: int
    payload: bytes
    observed_bits: int
    counter: int
    epoch: int
    calibration: bool | None


def parse_second_record(payload):
    if len(payload) < 13 or payload[:2] != b'\x08\x5d' or payload[-1] != 0:
        raise ValueError('Unexpected second-record wrapper')
    size = int.from_bytes(payload[2:4], 'little')
    if size % 8 or size != len(payload) - 5:
        raise ValueError('Inconsistent second-record length or entry size')
    return tuple(payload[i:i + 8] for i in range(4, 4 + size, 8))


def counter_from_record(payload):
    if len(payload) < 5 or payload[0] not in (1, 3):
        raise ValueError('Expected five header bytes in a type 01 or 03 record')
    return int.from_bytes(payload[1:5], 'little') >> COUNTER_SHIFT


def interior_records(frames, directory_rows, clocks):
    clock = {row['frame']: row for row in clocks}
    missing, records = [], []
    for row in directory_rows:
        frame = row['frame']
        state = clock[frame]
        words = frames[frame]
        if sorted(words) != row['available_words'] or not row['capacity_pass']:
            raise ValueError(f'Code 5 reception no longer matches E100 for frame {frame}')
        for number, (start, length) in enumerate(zip(row['starts'], row['lengths'], strict=True), 1):
            payload, observed = span(words, start, length)
            if payload is None:
                missing.append((frame, number, start, length))
                continue
            records.append(Record(frame, number, start, payload, observed, state['counter'],
                                  state['base_gps_s'], row.get('calibration')))
    return records, missing


def nonmonomial(poly):
    return poly >> ((poly & -poly).bit_length() - 1) if poly else 0


def crc_audit(blocks):
    return {orientation: {length: {'distinct': item['distinct'], 'difference_gcd': hex(item['gcd']),
                                   'nonmonomial_factor': hex(nonmonomial(item['gcd']))}
                          for length, item in difference_gcds(blocks, orientation).items()
                          if item['distinct'] >= MIN_DISTINCT_CRC}
            for orientation in ORIENTATIONS}


def compare_future(records):
    index = {}
    for record in records:
        if record.number >= 3 and record.payload[0] == 1:
            index.setdefault((record.epoch, counter_from_record(record.payload)), record)
    hist = {}
    for shift in PAIR_SHIFT:
        pairs = []
        for record in records:
            if record.number < 3 or record.payload[0] != 3:
                continue
            target = index.get((record.epoch, counter_from_record(record.payload) + shift))
            if target is None or record.frame == target.frame:
                continue
            prefix = (len(record.payload) >= PAIR_PREFIX_END and len(target.payload) >= PAIR_PREFIX_END
                      and record.payload[1:PAIR_PREFIX_END] == target.payload[1:PAIR_PREFIX_END])
            pairs.append({'source': [record.frame, record.number], 'target': [target.frame, target.number],
                          'both_calibration': record.calibration and target.calibration,
                          'both_held': record.calibration is False and target.calibration is False,
                          'same_length': len(record.payload) == len(target.payload), 'same_prefix': prefix,
                          'same_post_type': record.payload[1:] == target.payload[1:]})
        hist[str(shift)] = pairs
    return hist


def summarize(records, frames):
    second = {a.frame: parse_second_record(a.payload) for a in records if a.number == 2}
    rows, categories = [], defaultdict(list)
    for record in records:
        payload, number = record.payload, record.number
        category = f'position_{number}' if number < 3 else f'type_{payload[0]:02x}'
        categories[category].append(record)
        row = {'frame': record.frame, 'number': number, 'start': record.start,
               'length': len(payload), 'class': category, 'observed_bits': record.observed_bits,
               'calibration': record.calibration}
        if number == 2:
            row['entry_count'] = len(second[record.frame])
            row['wrapper_hash'] = hashlib.sha256(payload).hexdigest()
            row['same_first_32_bits_in_all_entries'] = len({entry[:4] for entry in second[record.frame]}) == 1
        elif category in ('type_01', 'type_03'):
            count = counter_from_record(payload)
            _, observed = span(frames[record.frame], record.start, 5)
            row.update(record_counter=count, first_record_counter=record.counter, delta=count - record.counter,
                       five_header_bytes_observed=observed == 40)
        elif category == 'type_02':
            field = int.from_bytes(payload[1:5], 'little')
            origin = int.from_bytes(second[record.frame][0][:4], 'little')
            row['delta_from_second_record_field'] = field - origin
        rows.append(row)
    crc = {category: crc_audit([record.payload for record in selected])
           for category, selected in sorted(categories.items())}
    return rows, crc


def narrow_coverage(windows, rows):
    outputs = []
    candidates = [r for r in rows if r['class'] in ('type_01', 'type_03')]
    for window in windows:
        max_word = max((r['start'] * 8 + COUNTER_RECORD_BIT + COUNTER_WIDTH - 1) // 1024 for r in candidates)
        systematic = [set(visible_positions(228 + 2048 * i, 1024, window['carriers'])) for i in range(max_word + 1)]
        hist = Counter()
        for r in candidates:
            start = r['start'] * 8 + COUNTER_RECORD_BIT
            available = sum((bit % 1024) in systematic[bit // 1024] for bit in range(start, start + COUNTER_WIDTH))
            hist[(r['class'], available)] += 1
        outputs.append({'kind': window['kind'], 'labels': window['labels'], 'direct_bits': {
            kind: {str(bits): n for (label, bits), n in sorted(hist.items()) if label == kind}
            for kind in ('type_01', 'type_03')}})
    return outputs


def load_evidence(artifact):
    old = 'v010-code5-parity-02'
    old_iq = artifact(old, 'E012', 'code5_observations.npz')
    with np.load(old_iq, allow_pickle=False) as src:
        generator = src['parity_generator']
    pn = equality_predictions(np.arange(1024), pn_bits(),
                              read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model'])
    first, second = archive_first_and_prior(artifact('v076-code5-byte-words', 'E078', 'archived_code5_words.npz'),
        artifact('v078-archive-code5-links', 'E080', 'recovered_code5_words.npz'), pn, generator)
    following_path = artifact('v089-code5-directory-final', 'E091', 'code5_directory_following_words.npz')
    with np.load(following_path, allow_pickle=False) as following:
        for i, data, mask in zip(following['frame_index'], following['bits'], following['observed'], strict=True):
            bits = np.unpackbits(data)
            if i in second or np.any(((bits[:1024] @ generator) & 1) != bits[1024:]):
                raise ValueError(f'Invalid stored second word: {i}')
            second[int(i)] = descramble(bits, pn), np.unpackbits(mask)[:1024].astype(bool)
    headers = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    counts = {i: quantity(row['h1']) for i, row in enumerate(headers)}
    archive, _, pages = recover_archive(counts, first, second, pn, generator)
    legacy_headers = read(artifact('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']
    legacy_counts = {r['legacy_frame_index']: quantity({**r['first_header'], 'layout20': r['first_layout20']})
                     for r in legacy_headers}
    ut = 'v022-ut-replay'
    ut_meta = read(artifact(ut, 'E014', 'ut_transfer.json'))
    ut_counts = {r['candidate_index']: quantity(r['h1']) for r in ut_meta['candidates']}
    return {'archive': archive, '480MSps': raw_words(old_iq, read(artifact(old, 'E012', 'code5_parity.json'))['words'],
             'hard_bits', legacy_counts, pn, generator), '250MSps': raw_words(artifact(ut, 'E014', 'ut_observations.npz'),
             ut_meta['words'], 'hard_code5_words', ut_counts, pn, generator)}, pages


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    boundary = read(artifact('v098-code5-records-final', 'E100', 'code5_directory_records.json'))
    clocks = read(artifact('v095-code5-clock-final', 'E097', 'code5_clock_correction.json'))
    frames, pages = load_evidence(artifact)
    if pages != boundary['archive_source']['pages']:
        raise ValueError('Qin hash and range receipts differ from E100')
    rows, missing, crc, future = {}, {}, {}, {}
    positives = []
    for dataset, code in frames.items():
        boundaries = boundary['archive'] if dataset == 'archive' else boundary['raw'][dataset]
        labels = clocks['archive'] if dataset == 'archive' else clocks['raw'][dataset]
        records, missing[dataset] = interior_records(code, boundaries, labels)
        rows[dataset], crc[dataset] = summarize(records, code)
        if dataset == 'archive':
            future = compare_future(records)
            positives = [code[r['frame']][0][0][:r['directory_length']] for r in boundaries]
        print(dataset, len(records), 'records', len(missing[dataset]), 'unavailable',
              'timing deltas', dict(Counter((r['class'], r['delta']) for r in rows[dataset] if 'delta' in r)))
    for shift, pairs in future.items():
        print('Future shift', shift, len(pairs), 'eligible', sum(p['same_length'] for p in pairs),
              'same length', sum(p['same_prefix'] for p in pairs), 'same prefix')
    outputs = {'schema': 'starlink-code5-later-records/v1',
               'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
               'archive_source': boundary['archive_source'], 'records': rows, 'unavailable_records': missing,
               'qin_future_pairs': future, 'record_crc_difference_gcds': crc, 'directory_crc_control': crc_audit(positives),
               'leo_counter_direct_projection': narrow_coverage(boundary['leo_windows'], rows['archive'])}
    output = experiment_output('E101')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'code5_later_records.json').write_text(json.dumps(outputs, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
