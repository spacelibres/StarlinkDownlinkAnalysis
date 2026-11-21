"""Audit shared record anchors, counters and length fields after channel decoding."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_mcs6_body_spans import RECORD_TAGS
from audit_mcs85_records import system_bytes
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


HEADER_BYTES = 20
ANCHOR_OFFSETS = np.array([0, 8, 12, 13, 17, 18, 19])
ANCHOR_VALUES = np.frombuffer(bytes.fromhex('103d006bb3ffd0'), dtype=np.uint8)
CONTROL_SHIFTS = (-2, -1, 1, 2)
SYSTEMATIC = {66: 12288, 70: 12800, 71: 10752, 75: 11008, 78: 11520, 81: 11776, 85: 12288, 88: 12800}


def inner_candidates(data, seen):
    if len(data) < HEADER_BYTES:
        return []
    w = sliding_window_view(data, HEADER_BYTES)
    mask = sliding_window_view(seen, HEADER_BYTES)
    match = ((w[:, ANCHOR_OFFSETS] == ANCHOR_VALUES) & mask[:, ANCHOR_OFFSETS]).all(axis=1)
    return np.flatnonzero(match).tolist()


def counter_pairs(headers):
    grouped = defaultdict(list)
    pairs = []
    for pos, word in headers:
        grouped[word[:6] + word[8:]].append((pos, word[6:8]))
    for identity, matches in grouped.items():
        for (a, x), (b, y) in zip(matches, matches[1:]):
            first = [int.from_bytes(x, endian) for endian in ('big', 'little')]
            last = [int.from_bytes(y, endian) for endian in ('big', 'little')]
            pairs.append({"first": a, "second": b, "common_header_without_counter_hex": identity.hex(),
                          "first_hex": x.hex(), "second_hex": y.hex(),
                          "big_endian_step": (last[0] - first[0]) % 65536,
                          "little_endian_step": (last[1] - first[1]) % 65536,
                          "big_endian_low_byte_wrap_on_plus_one": x[1] == 255 and y[1] == 0 and last[0] == (first[0] + 1) % 65536})
    return sorted(pairs, key=lambda row: row['first'])


def outer_record(data, start):
    if start < 0 or start + 4 > len(data) or data[start] not in RECORD_TAGS:
        return None
    field = int.from_bytes(data[start + 2:start + 4], 'little')
    span = 4 + (field >> 5)
    if field & 31 != 16 or span <= 4 or start + span > len(data):
        return None
    return span


def outer_chain(data, inner):
    records = []
    start = 0
    while (span := outer_record(data, start)) is not None:
        stop = start + span
        following = data[stop:]
        records.append({"start": start, "stop": stop, "outer_prefix_hex": data[start:start + 4].hex(),
                        "nested_inner_candidate": start + 4 in inner,
                        "zero_tail_bytes": len(following) if not any(following) else 0,
                        "next_known_type": bool(following and following[0] in RECORD_TAGS),
                        "shifted_next_type": {str(d): bool(0 <= stop + d < len(data) and data[stop + d] in RECORD_TAGS)
                                              for d in CONTROL_SHIFTS}})
        start = stop
    return records


def make_pn(model, shift=0):
    return equality_predictions(np.arange(max(SYSTEMATIC.values())), pn_bits(), {**model, 'phase': model['phase'] + shift})


def raw_streams(artifact, reserved, pn):
    two_names = read(artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_extraction.json'))['observed_allocations']
    five_names = read(artifact('v028-ut-cross32-extract-02', 'E030', 'ut_cross32_extraction.json'))['regions']
    by_frame = defaultdict(list)
    for row in two_names + five_names:
        by_frame[row['format'], row['frame']].append(row)
    run66 = 'v126-mcs66-full-validation'
    run70 = 'v129-mcs70-full-validation'
    run71 = 'v130-mcs-lower-rate-core'
    validation70 = read(artifact(run70, 'E131', 'mcs70_full_decoder_validation.json'))['measurements']
    validation71 = read(artifact(run71, 'E132', 'lower_rate_core.json'))['results']
    blocks = ((run66, 'E128', 'mcs66_full_decoder_words.npz', (66, 85), 'ordinary_bits'),
              (run70, 'E131', 'mcs70_full_decoded_words.npz', (70, 88), 'ordinary_bits'),
              (run71, 'E132', 'lower_rate_core_measured_words.npz', (71, 78), 'bits'))
    for run, exp, filename, codes, suffix in blocks:
        with np.load(artifact(run, exp, filename), allow_pickle=False) as z:
            for code in codes:
                packed = z[f'mcs{code}_{suffix}']
                if code in (66, 85):
                    frames = z[f'mcs{code}_frames']
                    if not z[f'mcs{code}_ordinary_pass'].all():
                        raise ValueError('Incomplete E128 channel decode')
                elif code in (70, 88):
                    frames = np.array([r['frame'] for r in validation70[str(code)]['rows']])
                else:
                    frames = np.array([r['frame'] for r in validation71[str(code)]['waveform']['rows']])
                    if not np.unpackbits(z[f'mcs{code}_seen'])[:SYSTEMATIC[code]].all():
                        raise ValueError('Partial core does not cover the full system area')
                if suffix == 'ordinary_bits' and not np.array_equal(packed, z[f'mcs{code}_masked_bits']):
                    raise ValueError('Raw codewords differ across the two decoder paths')
                data = system_bytes(np.unpackbits(packed, axis=1).astype(bool), SYSTEMATIC[code], pn)
                for frame in sorted(set(frames)):
                    found = by_frame[code, frame]
                    words = data[frames == frame]
                    if (len(found) != 1 or len(words) != found[0].get('count', found[0].get('quantity'))):
                        raise ValueError('Decoded bytes do not delimit an entire allocation')
                    yield code, int(frame), int(found[0]['start']), frame in reserved, words.ravel()


def archive_receipts(artifact, pn, wrong_pn):
    run = 'v120-cross-mcs-checks'
    archive = read(artifact(run, 'E122', 'cross_mcs_checks.json'))['archive']
    out = defaultdict(list)
    with np.load(artifact(run, 'E122', 'cross_mcs_observations.npz'), allow_pickle=False) as z:
        for code, k in SYSTEMATIC.items():
            for role in ('nonterminal', 'terminal'):
                marker = f'{code}_{role}'
                if marker not in archive:
                    continue
                rows = archive[marker]['rows']
                b = np.unpackbits(z[f'archive_{marker}_bits'], axis=1).astype(bool)
                m = np.unpackbits(z[f'archive_{marker}_seen'], axis=1).astype(bool)[:, :k]
                seen = m.reshape(len(m), -1, 8).all(axis=-1)
                normal = system_bytes(b, k, pn)
                wrong = system_bytes(b, k, wrong_pn)
                for row, data, control, mask in zip(rows, normal, wrong, seen, strict=True):
                    if row['calibration'] or row['group'] == 19:
                        continue
                    sites = inner_candidates(data, mask)
                    controls = inner_candidates(control, mask)
                    v = sliding_window_view(mask, HEADER_BYTES)[:, ANCHOR_OFFSETS].all(axis=1)
                    outer = outer_record(data.tobytes(), 0) if row['word'] == 0 and mask[:4].all() else None
                    out[code].append({"frame": row['frame'], "group": row['group'], "role": role,
                                      "word": row['word'], "resource": row['resource'], "eligible_starts": int(v.sum()),
                                      "inner_offsets": sites, "wrong_pn_offsets": controls,
                                      "outer_length_at_allocation_start": outer,
                                      "outer_nested_anchor": bool(outer and 4 in sites)})
    return dict(out)


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    labels = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    model = labels['frozen_pn_model']
    pn = make_pn(model)
    raw = defaultdict(list)
    terminal_padding = {(row['format'], row['frame']): row['start'] for row in labels['word_results']
                        if row['format'] in (78, 88) and row['allocation_final_word'] and row['accepted']}
    for code, frame, resource, reserved, bytes_in in raw_streams(artifact, labels['frame_split']['reserved'], pn):
        sites = inner_candidates(bytes_in, np.ones(len(bytes_in), dtype=bool))
        data = bytes_in.tobytes()
        headers = [(k, data[k:k + HEADER_BYTES]) for k in sites]
        raw[code].append({"frame": frame, "resource": resource, "receiver_reserved": bool(reserved),
                          "allocation_bytes": len(data), "records": [{"offset": k, "prefix_hex": h.hex()} for k, h in headers],
                          "counter_pairs": counter_pairs(headers), "outer_chain": outer_chain(data, set(sites)),
                          "independent_pn_start_bits": terminal_padding.get((code, frame))})
    archive = archive_receipts(artifact, pn, make_pn(model, 1))
    print('Raw formats', {k: (sum(len(r['records']) for r in v), sum(len(r['counter_pairs']) for r in v))
                          for k, v in raw.items()}, flush=True)
    print('Archive held groups with anchor', {k: len({r['group'] for r in v if r['inner_offsets']})
                                             for k, v in archive.items()}, flush=True)
    result = {"schema": "starlink-cross-mcs-record-fields/v1", "inner_offsets": ANCHOR_OFFSETS.tolist(),
              "inner_hex": ANCHOR_VALUES.tobytes().hex(), "anchor_discovery": "Retrospective same-capture prefix comparison",
              "raw": dict(raw), "qin_archive_held": archive, "excluded_archive_groups": [19],
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()}}
    out = experiment_output('E133')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'cross_mcs_record_fields.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
