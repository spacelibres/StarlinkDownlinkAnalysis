"""Compare sequence-field hypotheses on MCS 85 soft symbols."""

from itertools import pairwise
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from scipy.special import logsumexp

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_mcs85_records import SYSTEMATIC_ENDS, system_bytes
from audit_qin_lengths import SNAPSHOT, NONPILOT, pdu_stream, pn_reference
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, equality_predictions, observations, pn_bits, read, sort_points
from audit_ut_weak_frames import PREFIX


VALUES = np.arange(65536, dtype=np.int32)
VALUE_BITS = labels_as_bits(VALUES, 16)
LABEL_BITS = labels_as_bits(np.arange(32), 5)
REVERSE8 = labels_as_bits(np.arange(256), 8) @ (1 << np.arange(8))
ORDERS = {'be': VALUES, 'le': (VALUES >> 8) | ((VALUES & 255) << 8),
          'bit_reversed_be': REVERSE8[VALUES >> 8]*256 + REVERSE8[VALUES & 255]}
STEPS = tuple(range(-4, 5))
SCALES = (.03, .06)
NO_CARRY = (VALUES & 0xff00) | ((VALUES+1) & 0xff)
CARRY_STARTS = np.flatnonzero((VALUES & 255) == 255)
TAG_MAX_ERRORS = 2
IP_OFFSETS = range(65)
FCS_STARTS = range(33)
TRAILING = (0, 4, 8, 12, 16)
IPV6_NEXT_HEADERS = {0, 4, 6, 17, 41, 43, 44, 47, 50, 51, 58, 59, 60, 132, 135}


def field_distribution(symbols, points, pn, system_size, byte_offset, scale):
    """Sum constellation likelihoods over unconstrained bits of each symbol."""
    word, bit = np.divmod(byte_offset*8 + np.arange(16), system_size)
    _, starts = np.unique(np.column_stack((word, bit//5)), axis=0, return_index=True)
    score = np.zeros(len(VALUES))
    for start, stop in pairwise([*starts, 16]):
        distances = abs(symbols[word[start], bit[start]//5]-points)**2
        recovered_lanes = LABEL_BITS[:, bit[start:stop] % 5] ^ pn[bit[start:stop]]
        weights = 1 << np.arange(stop-start-1, -1, -1)
        subset = recovered_lanes @ weights
        bit_index = np.arange(start, stop)
        # Within each byte the first bit received is the least significant one.
        integer_bits = (bit_index//8)*8 + 7-bit_index % 8
        candidate_index = VALUE_BITS[:, integer_bits] @ weights
        log_weight = np.array([logsumexp(-distances[subset == label]/scale) for label in range(1 << (stop-start))])
        score += log_weight[candidate_index]
    return np.exp(score-logsumexp(score))


def tagged_pairs(rows, mapping):
    gap_tag = {gap: tag for tag, gap in mapping.items()}
    history = {}
    pairs = []
    for i, (a, b) in enumerate(pairwise(rows)):
        pos = a['byte_offset']
        gap = b['byte_offset']-pos
        tag = gap_tag.get(gap)
        if tag is None or (int(a['label_hex'], 16)^int(tag, 16)).bit_count() > TAG_MAX_ERRORS:
            history.clear()
            continue
        if tag in history:
            previous_index, previous_offset = history[tag]
            pairs.append({'tag': tag, 'from': previous_offset, 'to': pos, 'global_step': i-previous_index})
        history[tag] = i, pos
    return pairs


def step_score(p, q, step):
    return float(np.sum(p*np.roll(q, -step)))


def score_pair(p, q, ordinal):
    scores = {}
    for name, indices in ORDERS.items():
        first, second = p[indices], q[indices]
        scores[name] = {str(step): step_score(first, second, step) for step in STEPS}
    successor = (VALUES+1) % 65536
    pair = p*q[successor]
    carry = CARRY_STARTS
    return {'steps': scores, 'all_record_ordinal_score': step_score(p, q, ordinal),
            'all_record_ordinal_reverse_score': step_score(p, q, -ordinal),
            'individual_maxima': [int(p.argmax()), int(q.argmax())], 'plus_one_maximum_from': int(pair.argmax()),
            'carry_score': float(pair[carry].sum()), 'no_carry_score_on_wrap': float(np.dot(p[carry], q[NO_CARRY[carry]])),
            'no_carry_model_score': float(np.sum(p*q[NO_CARRY]))}


def internet_checksum_pass(header):
    carry = sum(int.from_bytes(header[i:i+2], 'big') for i in range(0, len(header), 2))
    while carry > 65535:
        carry = (carry >> 16)+(carry & 65535)
    return carry == 65535


def header_checks(data, start, stop):
    """Check plain byte-aligned headers and Ethernet FCS within a bounded record."""
    ipv4, ipv6, ethertype, fcs = [], [], [], []
    fragment = bytes(data[start:stop])
    for offset in IP_OFFSETS:
        pkt = fragment[offset:]
        if len(pkt) >= 20 and pkt[0] >> 4 == 4:
            ihl = (pkt[0] & 15)*4
            extent = int.from_bytes(pkt[2:4], 'big')
            if 20 <= ihl <= extent <= len(pkt) and not pkt[6] & 0x80 and pkt[8] and internet_checksum_pass(pkt[:ihl]):
                ipv4.append([offset, len(pkt)-extent])
        if len(pkt) >= 40 and pkt[0] >> 4 == 6 and pkt[6] in IPV6_NEXT_HEADERS and pkt[7]:
            extent = 40+int.from_bytes(pkt[4:6], 'big')
            if len(pkt)-extent in TRAILING:
                ipv6.append([offset, len(pkt)-extent])
        if len(pkt) >= 14 and pkt[12:14] in (b'\x08\x00', b'\x86\xdd', b'\x81\x00', b'\x88\xa8'):
            ethertype.append([offset, pkt[12:14].hex()])
    checked = 0
    for trim in TRAILING:
        end = len(fragment)-trim
        if end < 18:
            continue
        crc = int.from_bytes(fragment[end-4:end], 'little')
        for offset in FCS_STARTS:
            if end-offset < 18:
                break
            checked += 1
            if zlib.crc32(fragment[offset:end-4]) == crc:
                fcs.append([offset, trim])
    return {'ipv4_checksum_hits': ipv4, 'ipv4_extent_hits': [row for row in ipv4 if row[1] in TRAILING],
            'ipv6_extent_hits': ipv6, 'ethertype_hits': ethertype, 'ethernet_fcs_tested': checked, 'ethernet_fcs_hits': fcs}


def audit_frame(source, points, pn, raw, row, mapping):
    pairs = tagged_pairs(row['records'], mapping)
    by_position = {r['byte_offset']: r for r in row['records']}
    for pos, record in by_position.items():
        if raw[pos:pos+20].tobytes().hex() != record['observed_hex']:
            raise ValueError(f'Frozen record replay differs at frame {row["frame"]}, byte {pos}')
    for scale in SCALES:
        cache = {}
        for pair in pairs:
            a, b = pair['from'], pair['to']
            for pos in (a, b):
                if pos not in cache:
                    cache[pos] = field_distribution(source, points, pn, SYSTEMATIC_ENDS[85], pos+6, scale)
            pair.setdefault('scale', {})[str(scale)] = score_pair(cache[a], cache[b], pair['global_step'])
            pair['hard_values'] = [by_position[pos]['candidate_counter_be'] for pos in (a, b)]
    format_checks = []
    reversed_raw = REVERSE8[raw].astype(np.uint8)
    for link in row['length_predictions']:
        if link['status'] == 'matched':
            start, end = link['reference'], link['predicted']
            format_checks.append({'position': start, 'byte_count': end-start,
                                  'lsb_first': header_checks(raw, start, end),
                                  'msb_first': header_checks(reversed_raw, start, end)})
    return {'frame': row['frame'], 'pairs': pairs, 'format_checks': format_checks}


def qin_observations(source, row_by_frame, points, labels, inputs):
    group_source = read(checked_artifact(SNAPSHOT, 'E034', 'qin_snapshot_source.json', inputs))
    groups = group_source['hard_groups']
    paths = [checked_artifact(SNAPSHOT, 'E034', f'qin_hard_{g}.npz', inputs) for g in groups]
    phase = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    for frame, group, _, flat in pdu_stream(paths, groups, phase, PREFIX):
        if frame not in row_by_frame:
            continue
        row = row_by_frame[frame]
        first, count = row['resource_start'], row['candidate_words']
        width = 3277
        grid = (flat[first:first+count*width]*max(abs(points))).reshape(count, width)
        # Keep E052's point order for nearest-point ties in the archived decisions.
        bits = labels[nearest_symbols(grid, points)].reshape(count, -1)[:, :16384]
        mode = 'qin_overlap_group' if group in source['overlap_groups'] else 'qin_other_groups'
        yield grid, bits, row, mode


def main():
    require_run_context()
    inputs = {}
    def artifact(run, name):
        return read(checked_artifact(run, 'E052', name, inputs))
    records = artifact('v050-mcs85-spacing', 'mcs85_records.json')
    archive = artifact('v050-mcs85-qin', 'qin_record_transfer.json')
    datasets, split, pn_model = observations(inputs)
    if pn_model != records['pn_model'] or split != records['frames_split']:
        raise ValueError('E052 receiver configuration differs')
    pn = equality_predictions(np.arange(16384), pn_bits(), pn_model)
    width = SYSTEMATIC_ENDS[85]
    mapping = records['fitted_distances_by_exact_label']
    bits, group, grid, points = datasets[85]
    output = []
    for row in records['frame_results']:
        frame = row['frame']
        chosen = group == frame
        raw = system_bytes(bits[chosen], width, pn).ravel()
        result = audit_frame(grid[chosen], points, pn, raw, row, mapping)
        output.append({**result, 'partition': 'iq_calibration' if frame in split['calibration'] else 'iq_reserved'})
    by_frame = {row['frame']: row for row in archive['results']}
    if len(by_frame) != len(archive['results']):
        raise ValueError('Expected one Qin MCS 85 allocation per frame')
    qam = read(checked_artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json', inputs))
    qin_points = np.array([complex(r['i'], r['q']) for r in qam['cross32_mapping']])/np.sqrt(20)
    labels = labels_as_bits(np.array([r['label'] for r in qam['cross32_mapping']]), 5)
    soft_points = sort_points(qin_points, labels)
    for grid, bits, row, partition in qin_observations(archive, by_frame, qin_points, labels, inputs):
        raw = system_bytes(bits, width, pn).ravel()
        result = audit_frame(grid, soft_points, pn, raw, row, mapping)
        output.append({**result, 'partition': partition, 'archive_group': row['group']})
    result = {'schema': 'starlink-mcs85-counter/v1', 'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p,h in inputs.items()},
              'steps': STEPS, 'scales': SCALES, 'tag_max_hard_errors': TAG_MAX_ERRORS,
              'systematic_end_bits': width, 'frame_groups': split, 'pn_model': pn_model, 'length_map': mapping,
              'ip_offsets': [IP_OFFSETS.start, IP_OFFSETS.stop], 'fcs_offsets': [FCS_STARTS.start, FCS_STARTS.stop],
              'trailing_byte_controls': TRAILING, 'frames': output}
    path = experiment_output('E053')
    path.mkdir(parents=True, exist_ok=True)
    (path/'mcs85_counter.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Processed', len(output), 'frame/receiver observations;', sum(len(r['pairs']) for r in output), 'pairs', flush=True)


if __name__ == '__main__':
    main()
