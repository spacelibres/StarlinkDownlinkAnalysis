"""Read Code 5 record extents from their own packed fields."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_later_records import interior_records, load_evidence
from audit_h2_projection import visible_positions
from audit_ut_repeated_data import checked_artifact, read


HEADER_BYTES = 20
COUNT_BIT = 153
COUNT_WIDTH = 7
MARKER_ORIGIN = 161
MARKER = 0x100
MARKER_BITS = 12
FOOTER_LENGTH_WIDTH = 8
FOOTER_HEADER_BITS = 12
LEGACY_COUNT_BIT = 116
LEGACY_COUNT_WIDTH = 4


def read_extent(stream, *, unit_bits=60, count_shift=0, footer_shift=0, ignore_footer=False):
    if len(stream) < HEADER_BYTES or stream[0] not in (1, 3):
        return None
    tag = stream[17:19]
    if stream[0] == 1 and tag == b'\x60\x00' and stream[14] & 15 == 5:
        count = (int.from_bytes(stream[:16], 'little') >> (LEGACY_COUNT_BIT + count_shift)) & 15
        length = 16 + 7 * count
        if count and length <= len(stream):
            return {'format': 'legacy', 'count': count, 'bytes': length,
                    'fields': {'count': [LEGACY_COUNT_BIT + count_shift, LEGACY_COUNT_WIDTH]}}
        return None

    value = int.from_bytes(stream, 'little')
    count = (value >> (COUNT_BIT + count_shift)) & ((1 << COUNT_WIDTH) - 1)
    if not count and tag == b'\0\0' and not stream[19]:
        return {'format': 'empty', 'count': count, 'bytes': HEADER_BYTES,
                'fields': {'count': [COUNT_BIT + count_shift, COUNT_WIDTH]}}
    if not count or tag != b'\x10\xe0':
        return None
    marker_bit = MARKER_ORIGIN + unit_bits * count
    footer_length_bit = marker_bit + MARKER_BITS + 4 + footer_shift
    footer_start = marker_bit + MARKER_BITS + FOOTER_HEADER_BITS
    if footer_start > 8 * len(stream) or (value >> marker_bit) & 0xfff != MARKER:
        return None
    if value >> (marker_bit + MARKER_BITS) & 15:
        return None
    octets = 0 if ignore_footer else (value >> footer_length_bit) & 0xff
    end = footer_start + 8 * octets
    length = (end + 7) // 8
    if length > len(stream) or (value >> end) & ((1 << (8 * length - end)) - 1):
        return None
    return {'format': 'packed', 'count': count, 'marker_bit': marker_bit,
            'footer_octets': octets, 'padding_bits': 8 * length - end, 'bytes': length,
            'fields': {'count': [COUNT_BIT + count_shift, COUNT_WIDTH], 'marker': [marker_bit, MARKER_BITS],
                       'footer_length': [footer_length_bit, FOOTER_LENGTH_WIDTH]}}


def read_masked_extent(stream, available):
    """Read the length given a packed-layout start and a mask of observed bits."""
    value = int.from_bytes(stream, 'little')

    def field(bit, width):
        if bit + width > len(stream) * 8 or not all(available(i) for i in range(bit, bit + width)):
            return None
        return (value >> bit) & ((1 << width) - 1)

    count = field(COUNT_BIT, COUNT_WIDTH)
    if not count:
        return None
    marker_bit = MARKER_ORIGIN + 60 * count
    if field(marker_bit, MARKER_BITS) != MARKER:
        return None
    octets = field(marker_bit + MARKER_BITS + 4, FOOTER_LENGTH_WIDTH)
    if octets is None:
        return None
    return (marker_bit + MARKER_BITS + FOOTER_HEADER_BITS + 7) // 8 + octets


def buffer_at(words, start):
    index, offset = divmod(start, 128)
    stored = []
    while index in words:
        stored.append(words[index])
        index += 1
    stream = b''.join(w[0] for w in stored)
    mask = [b for word in stored for b in word[1]]
    return stream[offset:], mask[offset * 8:]


def direct_guards(mask, parsed):
    required = [(0, 8), (136, 16)] + list(parsed['fields'].values())
    if parsed['format'] == 'legacy':
        required.append((112, 4))
    elif parsed['format'] == 'empty':
        required.append((152, 8))
    else:
        required.extend([(parsed['marker_bit'] + MARKER_BITS, 4),
                         (parsed['bytes'] * 8 - parsed['padding_bits'], parsed['padding_bits'])])
    return all(all(mask[at:at + width]) and at + width <= len(mask) for at, width in required)


def field_coverage(windows, rows, words):
    max_byte = max(r['start'] + r['length'] for r in rows)
    packed = [(r, *buffer_at(words[r['frame']], r['start'])) for r in rows
              if r['parsed'] is not None and r['parsed']['format'] == 'packed']
    output = []
    for window in windows:
        carrier_rows = [set(visible_positions(228 + 2048 * k, 1024, window['carriers']))
                        for k in range((max_byte + 127) // 128)]
        fields, joint, replay = defaultdict(Counter), Counter(), []
        for row, stream, mask in packed:
            parsed = row['parsed']
            fully_visible = 0
            for field, (relative, width) in parsed['fields'].items():
                start = row['start'] * 8 + relative
                seen = sum(bit % 1024 in carrier_rows[bit // 1024] for bit in range(start, start + width))
                fields[field][seen] += 1
                fully_visible += seen == width
            joint[fully_visible] += 1
            record_origin = row['start'] * 8

            def available(relative):
                word, bit = divmod(record_origin + relative, 1024)
                return relative < len(mask) and mask[relative] and word < len(carrier_rows) and bit in carrier_rows[word]

            decision = read_masked_extent(stream, available)
            if decision is not None:
                replay.append({'frame': row['frame'], 'number': row['number'], 'predicted': decision,
                               'matches': decision == row['length']})
        output.append({'kind': window['kind'], 'labels': window['labels'], 'visible_bits': {k: dict(v) for k, v in fields.items()},
                       'joint_count_of_full_fields': dict(joint), 'measured_archive_replay': replay})
    return output


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    def load(run, exp, name):
        return read(artifact(run, exp, name))

    boundary = load('v098-code5-records-final', 'E100', 'code5_directory_records.json')
    clocks = load('v095-code5-clock-final', 'E097', 'code5_clock_correction.json')
    schedule = load('v136-code5-future-slots-final', 'E138', 'code5_future_slots.json')
    pairs = load('v138-code5-preview-records-masks', 'E140', 'code5_preview_records.json')['pair_predictions']
    partitions = {name: set(row['frames']) for name, row in schedule['qin'].items()}
    frames, pages = load_evidence(artifact)
    if pages != boundary['archive_source']['pages']:
        raise ValueError('Archive page offsets or fingerprints changed')
    all_rows, payloads = {}, {}
    summary = defaultdict(Counter)
    variants = {'count_one_bit_early': {'count_shift': -1}, 'count_one_bit_late': {'count_shift': 1},
                'stride_59': {'unit_bits': 59}, 'stride_61': {'unit_bits': 61},
                'footer_one_bit_early': {'footer_shift': -1}, 'footer_one_bit_late': {'footer_shift': 1},
                'omit_footer_length': {'ignore_footer': True}}
    controls = defaultdict(Counter)
    for label, words in frames.items():
        directories = boundary['archive'] if label == 'archive' else boundary['raw'][label]
        timing = clocks['archive'] if label == 'archive' else clocks['raw'][label]
        membership = {r['frame']: r for r in directories}
        records, missing = interior_records(words, directories, timing)
        by_record = {(r.frame, r.number): r for r in records}
        if label == 'archive':
            payloads = {key: r.payload for key, r in by_record.items()}
        rows = []
        for record in records:
            if record.number < 3 or record.payload[0] not in (1, 3):
                continue
            frame = record.frame
            group = next((name for name, selected in partitions.items() if frame in selected), 'other') if label == 'archive' else label
            stream, mask = buffer_at(words[frame], record.start)
            if stream[:len(record.payload)] != record.payload:
                raise ValueError('Contiguous Code 5 buffer differs from the independently bounded record')
            parsed = read_extent(stream)
            matches = parsed is not None and parsed['bytes'] == len(record.payload)
            n = record.number
            next_start = membership[frame]['starts'][n] if n < len(membership[frame]['starts']) else None
            next_matches = None if next_start is None else matches and record.start + parsed['bytes'] == next_start
            direct = bool(parsed is not None and direct_guards(mask, parsed))
            row = {'frame': frame, 'number': n, 'type': record.payload[0], 'start': record.start,
                   'length': len(record.payload), 'partition': group, 'parsed': parsed, 'length_matches': matches,
                   'header_fields_direct': direct, 'next_record_start': next_start, 'next_boundary_matches': next_matches,
                   'next_record_type': stream[parsed['bytes']] if next_matches and parsed['bytes'] < len(stream) else None}
            rows.append(row)
            bucket = summary[f'{label}/{group}']
            bucket['records'] += 1
            bucket['matching_extent'] += matches
            bucket['all_required_bits_direct'] += direct
            bucket['direct_matching_extent'] += direct and matches
            bucket['next_records'] += next_matches is not None
            bucket['next_boundaries_matched'] += bool(next_matches)
            if parsed is not None:
                bucket[parsed['format'] + '_records'] += 1
            if parsed is not None and parsed['format'] != 'empty':
                label_key = 'Qin_held' if label == 'archive' and group == 'holdout' else label
                for name, setting in variants.items():
                    if parsed['format'] == 'legacy' and 'count_' not in name:
                        continue
                    alternative = read_extent(stream, **setting)
                    controls[label_key][name + '_tested'] += 1
                    controls[label_key][name + '_matched'] += bool(alternative and alternative['bytes'] == len(record.payload))
        all_rows[label] = {'records': rows, 'missing': missing}

    transformed = []
    for pair in pairs:
        source = payloads[tuple(pair['source'])]
        shape = read_extent(source)
        if shape is None or shape['format'] != 'packed':
            raise ValueError('No intrinsic count available for the paired preview record')
        v = int.from_bytes(source, 'little') ^ 2
        for k in range(shape['count']):
            v &= ~(0xfff << (MARKER_ORIGIN + 60 * k))
        predicted = v.to_bytes(len(source), 'little')
        transformed.append({'source': pair['source'], 'target': pair['target'], 'same_axis': pair['same_axis'],
                            'count': shape['count'], 'matches': predicted == payloads[tuple(pair['target'])],
                            'matches_E140_digest': sha256(predicted).hexdigest() == pair['predicted_sha256']})

    result = {'schema': 'starlink-code5-intrinsic-extents/v1', 'input_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'archive_page_receipts': pages, 'summary': {k: dict(v) for k, v in summary.items()},
              'records': all_rows, 'wrong_field_controls': {k: dict(v) for k, v in controls.items()},
              'intrinsic_preview_conversion': transformed,
              'leo_direct_coverage': field_coverage(boundary['leo_windows'], all_rows['archive']['records'], frames['archive'])}
    for label, stats in result['summary'].items():
        print(label, stats)
    print('Intrinsic preview conversion', sum(r['matches'] and r['matches_E140_digest'] for r in transformed), '/', len(transformed))
    print('Controls', result['wrong_field_controls'])
    out = experiment_output('E141')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'code5_record_extents.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
