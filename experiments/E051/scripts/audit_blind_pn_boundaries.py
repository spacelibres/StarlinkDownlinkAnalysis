"""Locate PN15 padding stops on the whole grid before consulting H2."""

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import correlate

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, legacy_artifact, require_run_context
from audit_internal_allocations import KINDS, WIDTH, allocations, collapsed
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, fit_carrier_gain, load_samples, sha256
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from audit_ut_weak_frames import PREFIX
from code5_core import LOADED, equalize_grid, reference_gain
from receiver_models import pn_bits
from ut_receiver_core import capture_grid


SYSTEMATIC_ENDS = {57: 5376, 66: 12288, 70: 12800, 71: 10752,
                   78: 11520, 85: 12288, 88: 12800, 92: 13056}
WINDOW = 128
NEAR_WINDOW = 32
NEAR_AFTER_MAX = .75
BEFORE_MIN = .82
AFTER_MAX = .68
DROP_MIN = .20
CLUSTER_GAP = 16
TOLERANCE = 4
SEED = 510049


def sliding_match(values, template):
    match = correlate(values, template, mode='valid', method='fft').ravel()
    # Signed agreement is integral; remove FFT roundoff before comparing ties.
    return .5 + np.rint(match).astype(np.int32) / (2 * template.size)


def scan_reference(decisions, template, end_bits, word_width):
    """Yield any PN-to-non-PN change, in half-open nonpilot resource coordinates."""
    lanes = decisions.shape[1]
    stop, remainder = divmod(end_bits, lanes)
    skip = int(remainder != 0)
    values = 1 - 2 * decisions.astype(np.float32)
    signs = 1 - 2 * template.astype(np.float32)
    before = sliding_match(values, signs[stop-WINDOW:stop])
    after = sliding_match(values, signs[stop+skip:stop+skip+WINDOW])
    coordinates = np.arange(stop, len(values) - (word_width - stop) + 1)
    left = before[coordinates-WINDOW]
    right = after[coordinates+skip]
    hits = np.flatnonzero((left >= BEFORE_MIN) & (right <= AFTER_MAX) & (left-right >= DROP_MIN))
    if not len(hits):
        return []
    peaks = np.split(hits, np.flatnonzero(np.diff(hits) > CLUSTER_GAP) + 1)
    results = []
    for peak in peaks:
        best = peak[np.argmax(left[peak])]
        split = int(coordinates[best])
        after_start = split + skip
        near_match = np.mean(decisions[after_start:after_start+NEAR_WINDOW]
                             == template[stop+skip:stop+skip+NEAR_WINDOW])
        results.append({'first_non_systematic_or_mixed': split,
                        'word_start': split-stop, 'word_stop': split-stop+word_width,
                        'before_match': float(left[best]), 'after_match': float(right[best]),
                        'near_after_match': float(near_match),
                        'gated_sites': len(peak), 'peak_span': [int(coordinates[peak[0]]), int(coordinates[peak[-1]])]})
    return results


def hard_decisions(flat, points, labels):
    batches = np.array_split(flat, (len(flat) + 8191) // 8192)
    return np.concatenate([labels[np.argmin(abs(batch[:, None]-points)**2, axis=1)] for batch in batches])


def scan_frame(flat, alphabets, reference):
    decisions = {name: hard_decisions(flat, points, bits) for name, (points, bits) in alphabets.items()}
    order = np.random.default_rng(SEED).permutation(WINDOW//16)
    out = {'frozen_pn': [], 'shuffled_pn': []}
    for code, end in SYSTEMATIC_ENDS.items():
        observed = decisions[KINDS[code]]
        width = observed.shape[1]
        template = reference[:len(reference)//width*width].reshape(-1, width)
        cutoff = end//width
        wrong = template.copy()
        blocks = template[cutoff-WINDOW:cutoff].reshape(-1, 16, width)
        wrong[cutoff-WINDOW:cutoff] = blocks[order].reshape(WINDOW, width)
        for key, pn_template in (('frozen_pn', template), ('shuffled_pn', wrong)):
            matches = scan_reference(observed, pn_template, end, WIDTH[code])
            out[key].extend({'mcs': code, **match} for match in matches)
    return out


def matching_markers(code, stop, markers):
    return [i for i, marker in enumerate(markers)
            if marker['mcs'] == code and abs(marker['word_stop']-stop) <= TOLERANCE]


def compare_layout(first, entries, markers):
    body = allocations(first, entries)
    checks = []
    for index, (descriptor, (start, stop, kind)) in enumerate(zip(entries, body, strict=True)):
        code, count = descriptor['mcs_candidate'], descriptor['quantity']
        role = 'terminal' if index+1 == len(body) else ('same_alphabet' if kind == body[index+1][2] else 'changed_alphabet')
        matched = matching_markers(code, stop, markers)
        checks.append({'mcs': code, 'word_count': count, 'start': start, 'stop': stop, 'role': role,
                       'tested_marker': code in SYSTEMATIC_ENDS, 'matching_blind_marker_indices': matched})
    return body, checks


def ordering_controls(first, entries, markers, baseline):
    expected = collapsed(baseline)
    rows = []
    for i, j in combinations(range(len(entries)), 2):
        modified = entries.copy()
        modified[i], modified[j] = modified[j], modified[i]
        if modified == entries:
            continue
        layout, checks = compare_layout(first, modified, markers)
        rows.append({'swap': [i, j], 'same_alphabet_schedule': collapsed(layout) == expected,
                     'matching_allocation_count': sum(bool(c['matching_blind_marker_indices']) for c in checks),
                     'matching_blind_marker_indices': sorted({m for c in checks for m in c['matching_blind_marker_indices']})})
    return rows


def measured_grid(samples, frame, sss, words, decoded):
    raw, slots = capture_grid(samples, frame['pss']['epoch'], frame['sampling_scale'], frame['cfo_hz'])
    grid = equalize_grid(raw[1:], raw[0], sss, slots[1:])['nonpilot']
    grid /= reference_gain(grid.ravel()[:len(PREFIX)], PREFIX)
    good = [i for i, word in enumerate(words) if word['candidate_index'] == frame['candidate_index']
            and word['initial_parity_disagreements'] == 0]
    gain = fit_carrier_gain(grid, [words[i] for i in good], decoded[good])
    return (grid / gain[None, :]).ravel()


def main():
    require_run_context()
    hashes = {}

    def input_artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, hashes)

    capture = read(input_artifact('v022-ut-replay', 'E014', 'ut_transfer.json'))
    labels = read(input_artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    pn_model = read(input_artifact('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json'))['fit']['winner']
    reference = equality_predictions(np.arange(16384), pn_bits(), pn_model)
    with np.load(input_artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as source16:
        alphabets = {'16QAM': (source16['constellation_points'], source16['frozen_gray_mapping'])}
    points32 = np.array([complex(p['i'], p['q']) for p in labels['cross32_mapping']])/np.sqrt(20)
    bits32 = labels_as_bits(np.array([p['label'] for p in labels['cross32_mapping']]), 5)
    alphabets['cross32'] = points32, bits32

    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    sss_hash = sha256(sss_path)
    if sss_hash != capture['sss_reference_sha256']:
        raise ValueError('SSS reference changed')
    hashes[ROOT / 'references/qin/sssVec.mat'] = sss_hash
    samples = load_samples(ROOT / SOURCE, capture)
    hashes[ROOT / SOURCE] = capture['source_sha256']
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    with np.load(input_artifact('v022-ut-replay', 'E014', 'ut_observations.npz'), allow_pickle=False) as recorded:
        decoded = recorded['decoded_code5_words']
    rows = []
    for frame in capture['candidates']:
        if not frame['complete_frame']:
            continue
        flat = measured_grid(samples, frame, sss, capture['words'], decoded)
        coarse = scan_frame(flat, alphabets, reference)
        markers = {key: [m for m in group if m['near_after_match'] <= NEAR_AFTER_MAX]
                   for key, group in coarse.items()}
        h2 = frame['h2']
        if h2['crc_ok'] is not True:
            raise ValueError('Missing comparison header for a complete frame')
        first = frame['predicted_h2_start'] + 114*h2['blocks']
        entries = h2['layout20']['entries']
        body, checks = compare_layout(first, entries, markers['frozen_pn'])
        _, shuffled_checks = compare_layout(first, entries, markers['shuffled_pn'])
        rows.append({'frame': frame['candidate_index'], 'grid_resources': len(flat), 'markers': markers,
                     'coarse_markers': coarse,
                     'measured_layout_comparison': checks, 'shuffled_layout_comparison': shuffled_checks,
                     'ordering_controls': ordering_controls(first, entries, markers['frozen_pn'], body)})
        print('Frame', frame['candidate_index'], {k: len(v) for k, v in markers.items()}, flush=True)
    output = experiment_output('E051')
    output.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'starlink-blind-pn-boundaries/v1', 'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in hashes.items()},
              'expected_resampling_sha256': capture['resampling']['sha256'], 'pn_model': pn_model,
              'mapping_calibration_split': labels['frame_split'], 'systematic_end_bits': SYSTEMATIC_ENDS,
              'window_symbols': WINDOW, 'before_min': BEFORE_MIN, 'after_max': AFTER_MAX, 'drop_min': DROP_MIN,
              'near_after_window_symbols': NEAR_WINDOW, 'near_after_max': NEAR_AFTER_MAX,
              'cluster_gap_resources': CLUSTER_GAP, 'match_tolerance_resources': TOLERANCE,
              'shuffle_seed': SEED, 'shuffle_block_symbols': 16, 'frames': rows}
    (output/'blind_pn_boundaries.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
