"""Validate MCS 6 record lengths against observed headers and the padding onset."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_h2_projection import visible_positions
from audit_mcs6_directory_crc import LOW_MCS, directory_candidates, parse_compact_entries
from audit_qam_pn15 import equality_predictions
from audit_qin_mcs6_folded import K, prepare
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


RECORD_TAGS = frozenset((0x80, 0x90, 0xB8))


def span_from_entry(entry):
    if len(entry) != 5 or entry[0] & 15 != 3:
        raise ValueError('Expected a complete five-byte descriptor')
    return int.from_bytes(entry[2:5], 'little') >> 4


def body_spans(entries):
    if not entries:
        raise ValueError('Empty descriptor list')
    if entries[0] & 15 == 7:
        positions = (2,)
    elif len(entries) >= 7 and entries[0] & 15 == 3 and entries[5] & 15 == 7:
        positions = (0, 7)
    else:
        raise ValueError('Expected one optional unprefixed entry and a prefixed entry')
    result = [(i, span_from_entry(entries[i:i + 5])) for i in positions]
    if any(length < 4 for _, length in result):
        raise ValueError('Descriptor is shorter than a record header')
    return result


def quantity(frame):
    header = frame['h2']
    entries = [entry['quantity'] for entry in header['layout20']['entries'] if entry['mcs_candidate'] == 6]
    if not header['crc_ok'] or len(entries) != 1:
        raise ValueError('Expected one MCS 6 allocation in a CRC-checked H2')
    return entries[0]


def prefix_receipt(start, length, payload, erased, descriptor_offset):
    header = payload[start:start + 4]
    observed = not erased[start * 8:(start + 4) * 8].any()
    field = int.from_bytes(header[2:4], 'little')
    nested = (field >> 5) + 4 if field & 31 == 16 and observed and header[0] in RECORD_TAGS else None
    shifted = {}
    for delta in (-2, -1, 1, 2):
        position = start + delta
        if 0 <= position < len(payload) and not erased[position * 8:(position + 1) * 8].any():
            shifted[str(delta)] = payload[position] in RECORD_TAGS
    return {'descriptor_offset': descriptor_offset, 'start': start, 'length': length,
            'prefix': f'{header[0]:02x}', 'prefix_observed': not erased[start * 8:(start + 1) * 8].any(),
            'flag_low_5': field & 31 if observed else None, 'independent_length_when_flag_16': nested,
            'shifted_prefix_known_type': shifted}


def audit_frame(indices, metadata, reception, starts, observed, erased, crc, q):
    first, last = indices[0], indices[-1]
    frozen = reception[first]
    frame = frozen['frame']
    if any(metadata[i]['frame'] != frame for i in indices):
        raise ValueError('E036 and E103 frame order changed')
    if len(indices) != q or any(starts[i] != starts[first] + word * 16384 for word, i in enumerate(indices)):
        raise ValueError(f'Incomplete or discontinuous H2 allocation in frame {frame}')
    candidates = directory_candidates(observed[first], erased[first], crc)
    if len(candidates) != 1 or len(candidates[0]) != frozen['length']:
        raise ValueError(f'E103 directory cannot be reproduced in frame {frame}')
    header = candidates[0]
    if header[-3:].hex() != frozen['crc_hex'] or parse_compact_entries(header) != frozen['partition']:
        raise ValueError(f'E103 directory structure changed in frame {frame}')
    tail_offset = frozen['partition']['tail_offset']
    descriptors = body_spans(header[5:tail_offset])
    all_spans = [span_from_entry(header[i:i + 5]) for i, tag in zip(frozen['partition']['entry_offsets'],
                 frozen['partition']['entry_tags'], strict=True) if tag == 3]
    payload_bits = observed[indices].ravel()
    payload = np.packbits(payload_bits, bitorder='little').tobytes()
    payload_mask = erased[indices].ravel()
    pointers = []
    start = len(header)
    for offset, length in descriptors:
        if start + length > len(payload):
            raise ValueError(f'Predicted record exceeds the allocation in frame {frame}')
        pointers.append(prefix_receipt(start, length, payload, payload_mask, 5 + offset))
        start += length

    interval = metadata[last]['fixed']
    if not interval['accepted']:
        raise ValueError(f'Missing E036 independent terminal PN onset in frame {frame}')
    onset_bits = (len(indices) - 1) * K + interval['start']
    pad = payload_bits[start * 8:]
    seen = ~payload_mask[start * 8:]
    delta = start * 8 - onset_bits
    fixed_ends = [len(header) + sum(all_spans[:n]) for n in (1, 2)]
    match = {'frame': frame, 'group': frozen['group'], 'calibration': frozen['calibration'],
             'resource_start': int(starts[first]), 'directory_erasures': frozen['erased_coordinates'],
             'words_from_h2': q, 'capacity_bytes': len(payload), 'directory_bytes': len(header),
             'records': pointers, 'record_end_byte': start, 'onset_bits_e036': onset_bits,
             'end_minus_onset_bits': delta, 'round_onset_to_byte_delta': start - round(onset_bits / 8),
             'always_one_and_two_byte_errors': [p - round(onset_bits / 8) for p in fixed_ends],
             'padding_bits': len(pad), 'direct_padding_bits': int(seen.sum()),
             'observed_nonzero_padding_bits': int((seen & (pad != 0)).sum()),
             'remaining_unassigned_5_byte_descriptors': len(all_spans) - len(descriptors),
             'descriptor_length_field_low_nibbles': [header[p['descriptor_offset'] + 2] & 15 for p in pointers]}
    return match


def projection(windows, matches):
    output = []
    for row in windows:
        per_frame = []
        for frame in matches:
            start = frame['resource_start']
            selection = visible_positions(start + 5 * 8, 4, row['carriers'])
            counts = [len(visible_positions(start + r['descriptor_offset'] * 8 + 20, 20, row['carriers']))
                      for r in frame['records']]
            per_frame.append({'frame': frame['frame'], 'selection_bits': len(selection), 'length_bits': counts})
        output.append({'kind': row['kind'], 'labels': row['labels'], 'frames': per_frame})
    return output


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    audit = read(artifact(LOW_MCS, 'E036', 'qin_low_mcs_audit.json'))
    records = audit['formats']['6']['words']
    anchors = read(artifact('v036-qin-mcs6-transfer', 'E038', 'qin_mcs6_transfer.json'))['words']
    reference = read(artifact('v101-mcs6-directory', 'E103', 'mcs6_directory_crc.json'))
    headers = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    windows = read(artifact('v098-code5-records-final', 'E100', 'code5_directory_records.json'))['leo_windows']
    pn = equality_predictions(np.arange(16384), pn_bits(), audit['pn_model'])
    samples = artifact('v034-qin-low-mcs-extract', 'E036', 'qin_low_mcs_symbols.npz')
    with np.load(samples, allow_pickle=False) as z:
        source, frames, starts = z['mcs6_symbols'], z['mcs6_frames'], z['mcs6_resource_start']
    hard, mask = prepare(source, records, anchors, pn)['edge']
    values = hard[:, :K] ^ pn[:K]
    mask = mask[:, :K]
    by_frame = defaultdict(list)
    for i, frame in enumerate(frames):
        by_frame[int(frame)].append(i)
    results = [audit_frame(indices, records, reference['mcs6'], starts, values, mask,
                           reference['frozen_crc'], quantity(headers[frame])) for frame, indices in sorted(by_frame.items())]
    report = {'schema': 'starlink-mcs6-body-spans/v1', 'word_systematic_bits': K,
              'results': results, 'leo_direct_projection': projection(windows, results),
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    output = experiment_output('E104')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mcs6_body_spans.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for case in results:
        print(case['frame'], [r['length'] for r in case['records']], 'end', case['record_end_byte'],
              'difference from PN onset (bits)', case['end_minus_onset_bits'])


if __name__ == '__main__':
    main()
