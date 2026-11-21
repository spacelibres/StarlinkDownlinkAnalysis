"""Compare every same-constellation descriptor permutation using PN stop evidence."""

from itertools import groupby
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'learning')]
from project_paths import experiment_output, require_run_context
from audit_blind_pn_boundaries import (
    AFTER_MAX, BEFORE_MIN, DROP_MIN, NEAR_AFTER_MAX, NEAR_WINDOW, SYSTEMATIC_ENDS, WINDOW, hard_decisions,
)
from audit_full_tail_phase import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mixed_slot import TYPES
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex
from validate_code37 import constellation_for


SEED = 2026112169
MARKERS = {**SYSTEMATIC_ENDS, 6: 10240, 35: 9472, 42: 8704, 75: 11008, 81: 11776, 93: 13568}
KEYS = ('frozen', 'wrong_bit_phase', 'shuffled_before')


def padding_marker(before, after, before_ref, after_ref):
    observed = (after == after_ref)
    left = float((before == before_ref).mean())
    right = float(observed.mean())
    near = float(observed[:NEAR_WINDOW].mean())
    return {'before': left, 'after': right, 'near_after': near,
            'passed': left >= BEFORE_MIN and right <= AFTER_MAX and near <= NEAR_AFTER_MAX
                      and left-right >= DROP_MIN}


def ordering(edges):
    total, count = edges.shape
    ahead = np.zeros(total, dtype=np.int64)
    ways = np.zeros(total, dtype=np.int64)
    ways[-1] = 1
    positions = [set() for _ in range(count)]
    for mask in range(total-2, -1, -1):
        rows = [(int(edges[mask, k]+ahead[mask | 1 << k]), mask | 1 << k)
                for k in range(count) if not (mask & (1 << k))]
        ahead[mask] = max(s for s, _ in rows)
        ways[mask] = sum(ways[j] for s, j in rows if s == ahead[mask])
    reachable = {0}
    for mask in range(total-1):
        if mask not in reachable:
            continue
        for k in range(count):
            bit = 1 << k
            next_mask = mask | bit
            if not (mask & bit) and edges[mask, k]+ahead[next_mask] == ahead[mask]:
                positions[k].add(mask.bit_count())
                reachable.add(next_mask)
    original = int(sum(edges[(1 << k)-1, k] for k in range(count)))
    return {'original': original, 'best': int(ahead[0]), 'best_count': int(ways[0]),
            'possible_positions': [sorted(s) for s in positions],
            'original_order_optimal': bool(original == ahead[0]),
            'orders_compared': math.factorial(count)}


def expected_windows(code, bits_per_symbol, nominal, wrong, permutation):
    stop, partial = divmod(MARKERS[code], bits_per_symbol)
    next_symbol = stop + bool(partial)
    support = np.r_[np.arange(stop-WINDOW, stop), np.arange(next_symbol, next_symbol+WINDOW)]
    n, w = [sequence[:len(sequence)//bits_per_symbol*bits_per_symbol]
            .reshape(-1, bits_per_symbol)[support] for sequence in (nominal, wrong)]
    shuffled = n.copy()
    shuffled[:WINDOW] = n[:WINDOW].reshape(-1, 16, bits_per_symbol)[permutation].reshape(WINDOW, -1)
    return support, (n, w, shuffled)


def measure(flat, code, start, models, templates):
    model = TYPES[code]
    support, refs = templates[code]
    points, labels = models[model]
    data = flat[start+support]
    labels = hard_decisions(data, points, labels)
    return [padding_marker(labels[:WINDOW], labels[WINDOW:], expected[:WINDOW], expected[WINDOW:])
            for expected in refs]


def analyze_block(start, descriptors, flat, widths, models, templates):
    codes = [e['mcs_candidate'] for e in descriptors]
    span = [e['quantity']*widths[c] for e, c in zip(descriptors, codes, strict=True)]
    total = 1 << len(codes)
    prefix = np.zeros(total, dtype=np.int64)
    for m in range(1, total):
        bit = m & -m
        prefix[m] = prefix[m ^ bit]+span[bit.bit_length()-1]
    cache = {}
    passed = np.zeros((len(KEYS), total, len(codes)), dtype=np.int32)
    for mask in range(total-1):
        for i, code in enumerate(codes):
            if mask & (1 << i) or code not in MARKERS:
                continue
            final = start+int(prefix[mask])+span[i]-widths[code]
            key = code, final
            if key not in cache:
                cache[key] = measure(flat, code, final, models, templates)
            passed[:, mask, i] = [r['passed'] for r in cache[key]]
    ordered = [start+int(prefix[(1 << (i+1))-1]) for i in range(len(codes))]
    return {
        'start': start, 'stop': start+int(prefix[-1]), 'alphabet': TYPES[codes[0]], 'codes': codes,
        'ordered_ends': ordered, 'models': {k: ordering(passed[i]) for i, k in enumerate(KEYS)},
        'probes': [{'code': c, 'word_start': l, 'word_end': l+widths[c],
                    'scores': dict(zip(KEYS, s, strict=True))} for (c, l), s in cache.items()],
    }


def partitions(row, widths):
    first = row['h2_start']+114*row['h2']['blocks']
    for _, group in groupby(row['h2']['layout20']['entries'], key=lambda x: TYPES[x['mcs_candidate']]):
        run = list(group)
        if len(run) > 1 and any(e['mcs_candidate'] in MARKERS for e in run):
            yield first, run
        first += sum(widths[e['mcs_candidate']]*e['quantity'] for e in run)


def paired_transfer(records, overlap, raw):
    qin = {r['frame']: r['blocks'] for r in records}
    raw_frames = {r['frame']: r for r in raw['frames']}
    pairs = []
    for p in overlap['pairs']:
        if p['ut_candidate'] not in raw_frames:
            continue
        current = raw_frames[p['ut_candidate']]['measured_layout_comparison']
        probes = {(d['code'], d['word_end']): d['scores']['frozen'] for b in qin[p['qin_frame']] for d in b['probes']}
        for row in current:
            key = row['mcs'], row['stop']
            if row['role'] == 'same_alphabet' and key in probes:
                pairs.append({'qin_frame': p['qin_frame'], 'ut_candidate': p['ut_candidate'],
                              'code': key[0], 'word_end': key[1], 'archived_hard': probes[key],
                              'raw_passed': bool(row['matching_blind_marker_indices'])})
    return pairs


def main():
    require_run_context()
    inputs = {}
    def archive(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))
    headers = archive('v034-qin-header-survey', 'E036', 'qin_header_survey.json')['frames']
    layout = archive('v066-descriptor-layout-final', 'E068', 'descriptor_layout_identifiability.json')
    fit = next(c for c in layout['candidates'] if c['layout'] == layout['historical_h2_layout'])
    widths = dict(zip(fit['labels'], fit['absolute']['units'], strict=True))
    cross = archive('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')
    e051 = archive('v049-blind-pn-corrected', 'E051', 'blind_pn_boundaries.json')
    source16 = checked_artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz', inputs)
    with np.load(source16, allow_pickle=False) as p:
        qam16, labels16 = p['constellation_points'], p['frozen_gray_mapping']
    qam32 = np.array([complex(r['i'], r['q']) for r in cross['cross32_mapping']])
    labels32 = labels_as_bits(np.array([r['label'] for r in cross['cross32_mapping']]), 5)
    eight = constellation_for(2, False, True)
    models = {
        'BPSK': (np.array([1., -1.]), np.array([[0], [1]], dtype=np.uint8)),
        '8QAM': (eight/max(abs(eight)), labels_as_bits(np.arange(8), 3)),
        '16QAM': (qam16/max(abs(qam16)), labels16),
        'cross32': (qam32/max(abs(qam32)), labels32),
    }
    model = e051['pn_model']
    positions = np.arange(16384)
    pn = pn_bits()
    nominal = equality_predictions(positions, pn, model)
    wrong = equality_predictions(positions, pn, {**model, 'phase': model['phase']+1})
    permutation = np.random.default_rng(SEED).permutation(WINDOW//16)
    templates = {c: expected_windows(c, models[TYPES[c]][1].shape[1], nominal, wrong, permutation)
                 for c in MARKERS}
    slots = np.arange(2, 302)
    rotate = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    signs = 1-2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    results = []
    reader = CacheOnly(ROOT/'.cache/qin-mat-pages', False)
    with reader, h5py.File(reader, 'r') as data:
        hard = data['yDataDec']
        if hard.shape != (1009, 301, 1024):
            raise ValueError('Qin archive dimension changed')
        for a in range(0, hard.shape[0], hard.chunks[0]):
            b = min(a+hard.chunks[0], hard.shape[0])
            grid = as_complex(hard[a:b, 1:, :])[..., NONPILOT % 1024]
            for i, raw in enumerate(grid, a):
                header = headers[i]
                if header['frame'] != i or not (header['h1']['crc_ok'] and header['h2']['crc_ok']):
                    raise ValueError('Header identity or CRC changed')
                flat = remove_header_gain(raw*rotate, signs)
                blocks = [analyze_block(first, e, flat, widths, models, templates)
                          for first, e in partitions(header, widths)]
                results.append({'frame': i, 'group': a, 'held_of_unit_fit': a in layout['test_groups'], 'blocks': blocks})
            if a % 130 == 0 or b == 1009:
                print(f'Processed {b}/1009 Qin frames', flush=True)
    overlap = archive('v032-qin-exemplar-audit', 'E034', 'qin_exemplar_audit.json')['overlap']
    pairs = paired_transfer(results, overlap, e051)
    output = experiment_output('E069')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'same-alphabet-order/v1', 'seed': SEED, 'marker_systematic_ends': MARKERS,
              'shuffled_before_order': permutation.tolist(),
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
              'source': {'url': URL, 'etag': ETAG, 'bytes': SIZE,
                         'pages': [reader.receipts[k] for k in sorted(reader.receipts)]},
              'frames': results, 'raw_overlap_checks': pairs}
    (output/'same_alphabet_order.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
