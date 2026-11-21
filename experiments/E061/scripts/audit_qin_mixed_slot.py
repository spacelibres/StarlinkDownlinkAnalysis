"""Compare slot-level hard quantization with matched 250 MS/s IQ boundaries."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, legacy_artifact, require_run_context
from audit_blind_pn_boundaries import measured_grid
from audit_internal_allocations import ALPHABETS, KINDS
from audit_qin_lengths import regions_for
from audit_ut_binary_tail import SOURCE, cost, load_samples, sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from layout_core import NONPILOT
from snapshot_qin_exemplars import ArchiveReader, ETAG, PAGE, SIZE, URL, as_complex


WIDTH = len(NONPILOT)
CLASSES = ('16QAM', 'cross32')
TYPES = {**KINDS, 42: '16QAM', 75: 'cross32', 81: 'cross32', 93: 'cross32',
         6: 'BPSK', 35: '8QAM'}
MIN_SIDE = 64
LOCAL_WIDTH = 128
MATCH_WITHIN = 16
OFFSETS = (-128, -64, -32, 32, 64, 128)
SCALES = (.03, .06)
GAINS = np.arange(70, 141, 2)*.01
RING_TOLERANCE = 1e-6


def hard_rings(points):
    magnitude = abs(np.asarray(points))
    models = [abs(ALPHABETS[k])/max(abs(ALPHABETS[k])) for k in CLASSES]
    a, b = [(abs(magnitude[:, None]-p) <= RING_TOLERANCE).any(axis=1) for p in models]
    return {'16QAM_only': int((a & ~b).sum()), 'cross32_only': int((b & ~a).sum()),
            'shared': int((a & b).sum()), 'neither': int((~(a | b)).sum())}


def exclusive(stats):
    fits = [key for key in CLASSES if stats[f'{key}_only'] > 0]
    return fits[0] if len(fits) == 1 and stats['neither'] == 0 else None


def modulation_delta(values, left, right, left_gain, right_gain, variance):
    """Positive support for the first constellation under equal point priors."""
    a = cost(values, left_gain*ALPHABETS[left], variance)
    b = cost(values, right_gain*ALPHABETS[right], variance)
    return b-a


def split_evidence(scores, boundary):
    accumulated = np.r_[0., scores.cumsum()]
    locations = np.arange(MIN_SIDE, len(scores)-MIN_SIDE+1)
    best = locations[accumulated[locations].argmax()]
    hom = max(0., accumulated[-1])
    measures = {'best_resource_within_slot': int(best), 'best_gain_over_homogeneous': float(accumulated[best]-hom)}
    if boundary is not None:
        measures.update(
            at_predicted_gain_over_homogeneous=float(accumulated[boundary]-hom),
            preboundary_mean=float(scores[max(0, boundary-LOCAL_WIDTH):boundary].mean()),
            postboundary_mean=float(scores[boundary:boundary+LOCAL_WIDTH].mean()),
            shifted_gains={str(k): float(accumulated[boundary+k]-hom) for k in OFFSETS
                           if MIN_SIDE <= boundary+k <= len(scores)-MIN_SIDE},
        )
    return measures


def boundaries(row, widths):
    allocation = regions_for(row, widths)
    for left, right in zip(allocation, allocation[1:]):
        a, b = TYPES.get(left[0]), TYPES.get(right[0])
        slot, q = divmod(right[2], WIDTH)
        if {a, b} != set(CLASSES) or not MIN_SIDE <= q <= WIDTH-MIN_SIDE:
            continue
        left_earliest = (left[2]+WIDTH-1)//WIDTH
        right_latest = right[3]//WIDTH-1
        qualified = slot-2 >= left_earliest and slot+2 <= right_latest
        yield {'left_code': left[0], 'right_code': right[0], 'left_model': a, 'right_model': b,
               'resource': right[2], 'ofdm_slot': slot+2, 'within_slot': q,
               'calibration_slots': [slot, slot+4] if qualified else None,
               'control_slots': [slot+1, slot+3]}


def gain_for(calibration, kind, variance):
    selected = calibration[::7]
    scores = np.array([cost(selected, gain*ALPHABETS[kind], variance).mean() for gain in GAINS])
    return float(GAINS[scores.argmin()])


def audit_soft(grid, marker):
    slot = marker['ofdm_slot']-2
    q = marker['within_slot']
    left, right = marker['left_model'], marker['right_model']
    calibration = [grid[s-2] for s in marker['calibration_slots']]
    rows = []
    for variance in SCALES:
        a, b = [gain_for(data, kind, variance) for data, kind in zip(calibration, (left, right), strict=True)]
        deltas = [modulation_delta(grid[s], left, right, a, b, variance) for s in (slot, slot-1, slot+1)]
        mixed, *controls = [split_evidence(delta, boundary)
                            for delta, boundary in zip(deltas, (q, None, None), strict=True)]
        pure_max = max(r['best_gain_over_homogeneous'] for r in controls)
        comparison = max(0., pure_max, *mixed['shifted_gains'].values())
        success = (abs(mixed['best_resource_within_slot']-q) <= MATCH_WITHIN
                   and mixed['preboundary_mean'] > 0 and mixed['postboundary_mean'] < 0
                   and mixed['at_predicted_gain_over_homogeneous'] > comparison)
        rows.append({'variance': variance, 'amplitude_gains': [a, b], 'mixed_slot': mixed,
                     'pure_control_slots': controls, 'verified': success})
    return {'scales': rows, 'verified_both_scales': all(r['verified'] for r in rows)}


def audit_hard(frame, marker):
    slot = marker['ofdm_slot']-2
    q = marker['within_slot']
    values = frame[slot]
    before, after = hard_rings(values[max(0, q-LOCAL_WIDTH):q]), hard_rings(values[q:q+LOCAL_WIDTH])
    whole = hard_rings(values)
    pure = [hard_rings(frame[s-2]) for s in marker['control_slots']]
    return {'before': before, 'after': after, 'slot': whole, 'pure_controls': pure,
            'slot_exclusive_model': exclusive(whole),
            'both_sides_same_exclusive_model': exclusive(before) is not None and exclusive(before) == exclusive(after),
            'shows_expected_mixed_alphabets': exclusive(before) == marker['left_model']
                                             and exclusive(after) == marker['right_model']}


def coverage(headers, members):
    counts, words, occupied = Counter(), Counter(), Counter()
    for frame in sorted(members):
        present = set()
        for entry in headers[frame]['h2']['layout20']['entries']:
            code = entry['mcs_candidate']
            counts[code] += 1
            words[code] += entry['quantity']
            present.add(code)
        occupied.update(present)
    return {k: {'frames': occupied[k], 'allocations': counts[k], 'candidate_words': words[k]} for k in sorted(counts)}


class HardCacheReader(ArchiveReader):
    """Bind the extension to already archived HDF5 pages."""

    def load_page(self, start):
        stop = min(start+PAGE, SIZE)
        expected = self.cache/f'{start}-{stop-1}.bin'
        if not expected.is_file():
            raise FileNotFoundError(f'Required local archive page: {expected.name}')
        return super().load_page(start)


def hard_snapshot(paths):
    for path in paths:
        with np.load(path, allow_pickle=False) as store:
            for frame, values in zip(store['frame'], store['symbols'], strict=True):
                yield int(frame), values[1:, NONPILOT % 1024]


def hard_archive(file):
    data = file['yDataDec']
    if data.shape != (1009, 301, 1024):
        raise ValueError('Unexpected Qin hard archive shape')
    step = data.chunks[0]
    for start in range(0, len(data), step):
        records = as_complex(data[start:start+step])
        for frame, values in enumerate(records, start):
            yield frame, values[1:, NONPILOT % 1024]


def main():
    require_run_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-hard-cache', action='store_true')
    args = parser.parse_args()
    inputs = {}
    def checked(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)
    survey = read(checked('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))
    geometry = read(checked('v033-qin-lengths-final', 'E035', 'qin_lengths.json'))
    matched = read(checked('v032-qin-exemplar-audit', 'E034', 'qin_exemplar_audit.json'))
    snapshots = read(checked('v032-qin-snapshot', 'E034', 'qin_snapshot_source.json'))
    width = {int(k): v for k, v in {**geometry['inherited_units'], **geometry['fitted_units']}.items()}
    width[93] = 3277
    headers = {r['frame']: r for r in survey['frames']}
    snapshots_paths = []
    for group in snapshots['hard_groups']:
        snapshots_paths.append(checked('v032-qin-snapshot', 'E034', f'qin_hard_{group}.npz'))
    paired = {r['qin_frame']: r['ut_candidate'] for r in matched['overlap']['pairs']}
    capture = read(checked('v022-ut-replay', 'E014', 'ut_transfer.json'))
    with np.load(checked('v022-ut-replay', 'E014', 'ut_observations.npz'), allow_pickle=False) as source:
        known_words = source['decoded_code5_words']
    candidates = {r['candidate_index']: r for r in capture['candidates']}
    raw_path = ROOT/SOURCE
    raw = load_samples(raw_path, capture)
    inputs[raw_path] = capture['source_sha256']
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    reference_hash = sha256(sss_path)
    inputs[ROOT/'references/qin/sssVec.mat'] = reference_hash
    if reference_hash != capture['sss_reference_sha256']:
        raise ValueError('Unexpected synchronization reference')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    catalog_cut_count = sum(sum(1 for _ in boundaries(h, width)) for h in headers.values())
    raw_pairs = {}
    for frame, candidate_id in paired.items():
        base = headers[frame]
        candidate = candidates[candidate_id]
        if candidate['h2']['words_hex'] != base['h2']['words_hex']:
            raise ValueError('The raw/archive frame pairing has changed')
        markers = list(boundaries(base, width))
        grid = measured_grid(raw, candidate, sss, capture['words'], known_words).reshape(-1, WIDTH) if markers else None
        for marker in markers:
            row = {'ut_candidate': candidate_id, 'raw_last_complete_slot': len(grid)+1}
            if marker['calibration_slots'] is None:
                row['raw_status'] = 'too_few_pure_calibration_slots'
            elif max(marker['calibration_slots']) > row['raw_last_complete_slot']:
                row['raw_status'] = 'incomplete_raw_support'
            else:
                row['raw'] = audit_soft(grid, marker)
            raw_pairs[frame, marker['resource']] = row

    local = set()
    records = []
    def collect(observations):
        for frame, received in observations:
            local.add(frame)
            for marker in boundaries(headers[frame], width):
                key = (frame, marker['resource'])
                records.append({'frame': frame, **marker, 'hard': audit_hard(received, marker),
                                **raw_pairs.get(key, {})})

    archive = None
    if args.full_hard_cache:
        reader = HardCacheReader(ROOT/'.cache/qin-mat-pages', False)
        with reader, h5py.File(reader, "r") as stream:
            collect(hard_archive(stream))
        if len(records) != catalog_cut_count:
            raise ValueError('Full hard archive and header census disagree on candidate count')
        archive = {'url': URL, 'etag': ETAG, 'source_size': SIZE,
                   'pages': [reader.receipts[k] for k in sorted(reader.receipts)]}
    else:
        collect(hard_snapshot(snapshots_paths))
    result = {
        'schema': 'starlink-qin-mixed-slot/v1',
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        'classification_margin_each_side': MIN_SIDE, 'raw_cut_tolerance': MATCH_WITHIN,
        'catalog_frames': len(headers), 'hard_frames': len(local), 'paired_raw_frames': len(paired),
        'catalog_predicted_cut_count': catalog_cut_count, 'full_hard_archive': archive,
        'coverage': {name: coverage(headers, group) for name, group in
                     (('catalog', headers), ('hard', local), ('paired', paired))},
        'records': records,
    }
    out = experiment_output('E061')
    out.mkdir(parents=True, exist_ok=True)
    (out/'qin_mixed_slot.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Boundary previews in catalog / hard / paired:', catalog_cut_count, len(records),
          sum('raw' in row for row in records), flush=True)


if __name__ == '__main__':
    main()
