"""Measure strong PN15 regions at every frozen QAM word position."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'learning')]
from project_paths import experiment_output, require_run_context
from audit_full_tail_phase import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mixed_slot import TYPES
from audit_same_alphabet_order import MARKERS
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


SEED = 2026112170
WINDOW = 512
GRIDS = (1, 2, 4, 8, 16, 32, 64)
MIN_START = 128
LATTICE_TOLERANCE = 2e-6


def lattice_model(points, labels):
    points = points/max(abs(points))
    nonzero = np.r_[abs(points.real), abs(points.imag)]
    step = float(nonzero[nonzero > LATTICE_TOLERANCE].min())
    coords = np.rint(np.column_stack((points.real, points.imag))/step).astype(int)
    limit = int(abs(coords).max())
    mapping = np.full((2*limit+1, 2*limit+1), -1, dtype=int)
    mapping[coords[:, 0]+limit, coords[:, 1]+limit] = np.arange(len(points))
    return step, limit, mapping, labels


def demap_exact(samples, model):
    step, limit, mapping, labels = model
    finite = np.isfinite(samples)
    complex_grid = samples[finite]/step
    real = np.rint(complex_grid.real)
    imag = np.rint(complex_grid.imag)
    near = ((abs(real) <= limit) & (abs(imag) <= limit) &
            (abs(complex_grid-(real+1j*imag))*step <= LATTICE_TOLERANCE))
    integer_r, integer_i = real[near].astype(int)+limit, imag[near].astype(int)+limit
    points = np.full(len(samples), -1, dtype=int)
    points[np.flatnonzero(finite)[near]] = mapping[integer_r, integer_i]
    return labels[np.maximum(points, 0)], points >= 0


def word_windows(bits, valid, reference, stop):
    selected = slice(stop-WINDOW, stop+WINDOW)
    agrees = (bits[..., selected] == reference[selected]) & valid[..., selected]
    known = valid[..., selected].reshape(-1, 2, WINDOW).sum(axis=-1)
    hits = agrees.reshape(-1, 2, WINDOW).sum(axis=-1)
    rate = hits/np.maximum(known, 1)
    observed = (known >= .9*WINDOW).all(axis=1)
    before, after = rate.T
    passing = observed & (before >= .85) & (after <= .68) & (before-after >= .20)
    return {'available': observed, 'passed': passing, 'known': known, 'hits': hits}


def grid_evidence(scores):
    n = np.arange(len(scores))+MIN_START
    term = np.exp(scores-scores.max())
    base = term.mean()
    results = []
    for step in GRIDS:
        counts = np.bincount(n % step, minlength=step)
        weights = np.bincount(n % step, weights=term, minlength=step)
        results.append(np.log(np.maximum(weights/counts, 1e-300)/base).tolist())
    return results


def prefix_profile(bits, known, reference, stop):
    match = (bits[:stop] == reference[:stop])
    score = np.where(known[:stop], np.where(match, np.log(.97/.5), np.log(.03/.5)), 0)
    suffix = np.cumsum(score[::-1])[::-1][:stop-WINDOW+1]
    pick = int(suffix.argmax())
    near = np.flatnonzero(suffix >= suffix[pick]-2.)
    support = int(known[pick:stop].sum())
    errors = support-int((known[pick:stop] & match[pick:stop]).sum())
    return {'ml_start': pick, 'two_nat_minmax': [int(near[0]), int(near[-1])],
            'known_bits_from_ml': support, 'differences_from_ml': errors,
            'grid_log_evidence': grid_evidence(suffix[MIN_START:]) if pick >= MIN_START else None}


def allocation(flat, first, label, count, width, alphabet, nominal, wrong, shuffle):
    stop = MARKERS[label]
    hard, seen = demap_exact(flat[first:first+count*width], alphabet)
    bps = hard.shape[1]
    bits = hard.reshape(count, width*bps)
    valid = np.repeat(seen, bps).reshape(bits.shape)
    n = len(bits[0])
    reference = nominal[:n]
    alternative = reference.copy()
    before = reference[stop-WINDOW:stop].reshape(-1, 32)
    alternative[stop-WINDOW:stop] = before[shuffle].ravel()
    result, wrong_phase, shuffled = [word_windows(bits, valid, r[:n], stop)
                                    for r in (reference, wrong, alternative)]
    passing = []
    for i in np.flatnonzero(result['passed']):
        row = prefix_profile(bits[i], valid[i], reference, stop)
        row.update(word=int(i), resource_start=first+int(i)*width, last=bool(i == count-1),
                   boundary_known=result['known'][i].tolist(), boundary_hits=result['hits'][i].tolist())
        passing.append(row)
    return {'label': label, 'start': first, 'count': count, 'width_resources': width, 'bits_per_symbol': bps,
            'eligible_words': np.flatnonzero(result['available']).tolist(),
            'passed_wrong_phase': np.flatnonzero(wrong_phase['passed']).tolist(),
            'passed_shuffled_before': np.flatnonzero(shuffled['passed']).tolist(), 'strong_words': passing}


def summarize_alignment(frames):
    by_group = defaultdict(list)
    for frame in frames:
        for row in frame['allocations']:
            for w in row['strong_words']:
                if w['grid_log_evidence'] is not None and w['last']:
                    by_group[row['label'], frame['group'], frame['held']].append(w['grid_log_evidence'])
    results = {}
    for label in sorted({key[0] for key in by_group}):
        learned = [v for (c, _, held), v in by_group.items() if c == label and not held]
        reserved = [v for (c, _, held), v in by_group.items() if c == label and held]
        if not learned or not reserved:
            continue
        arms = []
        for i, step in enumerate(GRIDS):
            cal_mean = np.mean([np.mean([row[i] for row in group], axis=0) for group in learned], axis=0)
            held_mean = np.mean([np.mean([row[i] for row in group], axis=0) for group in reserved], axis=0)
            phase = int(cal_mean.argmax())
            arms.append({'grid': step, 'calibration_selected_remainder': phase,
                         'calibration_mean': cal_mean.tolist(), 'held_mean': held_mean.tolist(),
                         'held_at_fixed_remainder': float(held_mean[phase])})
        results[str(label)] = {'calibration_groups': len(learned), 'reserved_groups': len(reserved),
                               'calibration_words': sum(map(len, learned)), 'reserved_words': sum(map(len, reserved)),
                               'learned_grid': max(arms, key=lambda r: r['calibration_mean'][r['calibration_selected_remainder']])['grid'],
                               'arms': arms}
    return results


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
    source16 = checked_artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz', inputs)
    with np.load(source16, allow_pickle=False) as p:
        model16 = lattice_model(p['constellation_points'], p['frozen_gray_mapping'])
    points32 = np.array([complex(v['i'], v['q']) for v in cross['cross32_mapping']])
    labels32 = labels_as_bits(np.array([v['label'] for v in cross['cross32_mapping']]), 5)
    demappers = {'16QAM': model16, 'cross32': lattice_model(points32, labels32)}
    pn = pn_bits()
    model = cross['frozen_pn_model']
    positions = np.arange(16385)
    nominal = equality_predictions(positions, pn, model)
    wrong = equality_predictions(positions, pn, {**model, 'phase': model['phase']+1})
    shuffle = np.random.default_rng(SEED).permutation(WINDOW//32)
    slots = np.arange(2, 302)
    rotate = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    signs = 1-2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    frames = []
    reader = CacheOnly(ROOT/'.cache/qin-mat-pages', False)
    with reader, h5py.File(reader, 'r') as file:
        source = file['yDataDec']
        if source.shape != (1009, 301, 1024):
            raise ValueError('Qin archive shape changed')
        for a in range(0, source.shape[0], source.chunks[0]):
            b = min(a+source.chunks[0], source.shape[0])
            data = as_complex(source[a:b, 1:, :])[..., NONPILOT % 1024]
            for index, grid in enumerate(data, a):
                h = headers[index]
                if h['frame'] != index or not (h['h2']['crc_ok'] and h['h1']['crc_ok']):
                    raise ValueError('Header ID or CRC changed')
                start = h['h2_start']+114*h['h2']['blocks']
                flat = remove_header_gain(grid*rotate, signs)
                rows = []
                for entry in h['h2']['layout20']['entries']:
                    label, count = entry['mcs_candidate'], entry['quantity']
                    width = widths[label]
                    if label in MARKERS and TYPES[label] in demappers:
                        rows.append(allocation(flat, start, label, count, width, demappers[TYPES[label]], nominal, wrong,
                                               shuffle))
                    start += count*width
                frames.append({'frame': index, 'group': a, 'held': a in layout['test_groups'], 'allocations': rows})
            if a % 130 == 0 or b == 1009:
                print('Measured padding coordinates', b, 'of', 1009, flush=True)
    output = experiment_output('E070')
    output.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'pn-padding-starts/v1', 'seed': SEED, 'control_permutation': shuffle.tolist(),
              'grids': GRIDS, 'min_start_for_alignment': MIN_START, 'marker_systematic_ends': MARKERS,
              'input_hashes': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
              'source': {'url': URL, 'etag': ETAG, 'bytes': SIZE,
                         'pages': [reader.receipts[k] for k in sorted(reader.receipts)]},
              'word_windows_and_onsets': frames, 'grid_evidence_by_code': summarize_alignment(frames)}
    (output/'pn_padding_starts.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
