"""Decode packed child lengths and validate Starlink outer-record boundaries."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_directory_checks import crc_bytes
from audit_cross_mcs_record_fields import (ANCHOR_OFFSETS, ANCHOR_VALUES, SYSTEMATIC, make_pn, raw_streams)
from audit_mcs6_body_spans import RECORD_TAGS
from audit_mcs6_directory_crc import LOW_MCS, parse_compact_entries
from audit_mcs85_records import system_bytes
from audit_qam_pn15 import equality_predictions
from audit_qin_mcs6_folded import K, prepare
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


CONTROLS = (0, -2, -1, 1, 2)


def read_bits(data, start, width, seen):
    stop = start + width
    if stop > 8 * len(data) or (seen is not None and not seen[start:stop].all()):
        return None
    byte, shift = divmod(start, 8)
    return (int.from_bytes(data[byte:(stop + 7) // 8], 'little') >> shift) & ((1 << width) - 1)


def parse_outer(data, start, seen=None, allow_extent=False):
    if start < 0 or start + 4 > len(data) or read_bits(data, 8 * start, 8, seen) not in RECORD_TAGS:
        return None
    field = read_bits(data, 8 * (start + 2), 16, seen)
    if field is None or field >> 5 == 0:
        return None
    lengths = [field >> 5]
    terminal = bool(field & 16)
    pos = 8 * (start + 4)
    while not terminal:
        descriptor = read_bits(data, pos, 12, seen)
        if descriptor is None or descriptor >> 1 == 0:
            return None
        lengths.append(descriptor >> 1)
        terminal = bool(descriptor & 1)
        pos += 12
    payload_start = (pos + 7) // 8
    stop = payload_start + sum(lengths)
    if stop > len(data) and not allow_extent:
        return None
    starts = [payload_start]
    for length in lengths[:-1]:
        starts.append(starts[-1] + length)
    fill = payload_start * 8 - pos
    return {'start': start, 'stop': stop, 'type_hex': f'{data[start]:02x}', 'first_length_low_nibble': field & 15,
            'header_bytes': payload_start - start, 'child_lengths': lengths, 'child_starts': starts,
            'header_padding': read_bits(data, pos, fill, seen) if fill else None}


def certified_directory(data, crc):
    if len(data) < 12:
        return None
    length = data[0]
    if data[1] or length % 4 or not 12 <= length <= min(252, len(data)):
        return None
    header = data[:length]
    if crc_bytes(header[:-3], *crc) != header[-3:]:
        return None
    grammar = parse_compact_entries(header)
    if not grammar['tail_and_filler_fit'] or not grammar['two_byte_entries_precede_five']:
        return None
    return {'length': length, 'crc_hex': header[-3:].hex(), 'partition': grammar}


def directory_reference(artifact):
    audit = read(artifact(LOW_MCS, 'E036', 'qin_low_mcs_audit.json'))
    reference = read(artifact('v102-mcs6-spans', 'E104', 'mcs6_body_spans.json'))['results']
    anchors = read(artifact('v036-qin-mcs6-transfer', 'E038', 'qin_mcs6_transfer.json'))['words']
    pn = equality_predictions(np.arange(16384), pn_bits(), audit['pn_model'])
    with np.load(artifact('v034-qin-low-mcs-extract', 'E036', 'qin_low_mcs_symbols.npz'), allow_pickle=False) as z:
        candidates, frames = z['mcs6_symbols'], z['mcs6_frames']
    hard, erased = prepare(candidates, audit['formats']['6']['words'], anchors, pn)['edge']
    values = hard[:, :K] ^ pn[:K]
    report = []
    for r in reference:
        take = frames == r['frame']
        payload = np.packbits(values[take].ravel(), bitorder='little').tobytes()
        seen = ~erased[take, :K].ravel()
        rows = []
        for entry in r['records']:
            start, length = entry['start'], entry['length']
            parsed = parse_outer(payload, start, seen)
            if parsed is None or parsed['stop'] != start + length:
                raise ValueError(f"MCS 6 directory length disagrees in frame {r['frame']}")
            rows.append({**parsed, 'directory_length': length,
                         'observed_header_hex': payload[start:start + parsed['header_bytes']].hex()})
        report.append({'frame': r['frame'], 'group': r['group'], 'historical_calibration': r['calibration'], 'records': rows})
    return report


def wideband(artifact, labels, crc):
    earlier = read(artifact('v131-cross-mcs-records', 'E133', 'cross_mcs_record_fields.json'))['raw']
    records = []
    for code, frame, resource, reserved, values in raw_streams(artifact, labels['frame_split']['reserved'],
                                                            make_pn(labels['frozen_pn_model'])):
        data = values.tobytes()
        reference = next(r for r in earlier[str(code)] if r['frame'] == frame and r['resource'] == resource)
        anchors = {r['offset'] for r in reference['records']}
        directory = certified_directory(data, crc)
        at = directory['length'] if directory else 0
        packets, starts = [], set()
        while (record := parse_outer(data, at)) is not None:
            record['child_has_anchor'] = [j in anchors for j in record['child_starts']]
            starts.update(record['child_starts'])
            packets.append(record)
            at = record['stop']
        if not packets or any(data[at:]) or not anchors <= starts:
            raise ValueError(f"Raw MCS {code} frame {frame} did not close on an independently observed boundary")
        records.append({'mcs': code, 'frame': frame, 'resource': resource, 'receiver_reserved': bool(reserved),
                        'directory': directory, 'records': packets, 'end': at, 'capacity': len(data), 'zero_tail_bytes': len(data) - at,
                        'anchor_count': len(anchors), 'child_start_controls': {str(d): sum(p + d in anchors for p in starts)
                                                                              for d in CONTROLS}})
    return records


def archive_predictions(artifact, pn):
    run = 'v120-cross-mcs-checks'
    metadata = read(artifact(run, 'E122', 'cross_mcs_checks.json'))['archive']
    receipts = []
    with np.load(artifact(run, 'E122', 'cross_mcs_observations.npz'), allow_pickle=False) as source:
        for code, cutoff in SYSTEMATIC.items():
            for role in ('nonterminal', 'terminal'):
                key = f'{code}_{role}'
                if key not in metadata:
                    continue
                rows = metadata[key]['rows']
                first = np.array([r['word'] == 0 and not r['calibration'] and r['group'] != 19 for r in rows])
                raw = np.unpackbits(source[f'archive_{key}_bits'][first], axis=1).astype(bool)
                seen = np.unpackbits(source[f'archive_{key}_seen'][first], axis=1)[:, :cutoff].astype(bool)
                data = system_bytes(raw, cutoff, pn)
                for row, rx, mask in zip((r for r, use in zip(rows, first, strict=True) if use), data, seen, strict=True):
                    parsed = parse_outer(rx.tobytes(), 0, mask, allow_extent=True)
                    if parsed is None:
                        continue
                    present = mask.reshape(-1, 8).all(axis=1)
                    attempts = []
                    for start in parsed['child_starts']:
                        coords = start + np.asarray(CONTROLS)[:, None] + ANCHOR_OFFSETS[None]
                        if coords.min() < 0 or coords.max() >= len(rx) or not present[coords].all():
                            continue
                        hits = (rx[coords] == ANCHOR_VALUES[None]).all(axis=1)
                        attempts.append({'start': start, 'matches': hits.tolist()})
                    receipts.append({'mcs': code, 'frame': row['frame'], 'group': row['group'], 'resource': row['resource'],
                                     'parsed_from_start': parsed, 'within_word': attempts})
    return receipts


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    crc = model['crc_polynomial_including_top_bit'], model['affine_convention']['seed'], model['affine_convention']['residue']
    labels = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    low = directory_reference(artifact)
    raw = wideband(artifact, labels, crc)
    archive = archive_predictions(artifact, make_pn(labels['frozen_pn_model']))
    tot = np.asarray([r['matches'] for x in archive for r in x['within_word']]).sum(axis=0)
    document = {'schema': 'starlink-packed-outer-record-lengths/v1', 'archive_anchor_shifts': CONTROLS,
                'archive_anchor_match_counts': tot.tolist(), 'archive_excludes_calibration_and_group': [19],
                'mcs6_crc_directory_reference': low, 'wideband': raw, 'qin_archive_candidates': archive,
                'inputs_sha256': {p.relative_to(ROOT).as_posix(): v for p, v in inputs.items()}}
    out = experiment_output('E134')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'outer_record_lengths.json').write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('MCS 6 CRC-directory extents', sum(len(x['records']) for x in low), flush=True)
    print('Raw', len(raw), 'allocations,', sum(len(x['records']) for x in raw), 'outer records,',
          sum(len(y['child_lengths']) for x in raw for y in x['records']), 'children and',
          sum(x['anchor_count'] for x in raw), 'anchor matches', flush=True)
    print('Archive', len(archive), 'headers,', sum(len(x['within_word']) for x in archive), 'shared-mask predictions,', tot, flush=True)


if __name__ == '__main__':
    main()
