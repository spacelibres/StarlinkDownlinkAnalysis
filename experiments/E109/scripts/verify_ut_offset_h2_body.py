"""Check the frozen weak H2 hypothesis against previously unused raw body resources."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS, WINDOW
from audit_internal_allocations import KINDS, WIDTH
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_offset_constrained_h2 import local_prefix_phase
from audit_ut_offset_grid_pilots import PILOT, PN, SMOOTH_BINS, read_grid
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from receiver_models import pn_bits


SEED = 20261121110
HALF_WIDTH = 800
CONTROL_COUNT = 63
MODULATION_BITS = {'16QAM': 4, 'cross32': 5}


def predicted_markers(start, entries):
    markers = []
    for entry in entries:
        code, count = entry['mcs_candidate'], entry['quantity']
        if count <= 0:
            raise ValueError('Expected positive allocation quantity')
        width = WIDTH[code]
        stop = start + count * width
        bit_width = MODULATION_BITS[KINDS[code]]
        full, mixed = divmod(SYSTEMATIC_ENDS[code], bit_width)
        markers.append({'mcs': code, 'count': count, 'allocation_start': start, 'word_stop': stop,
                        'full_systematic_stop': stop - width + full, 'full_systematic_symbols': full,
                        'systematic_mixed_skip': int(mixed != 0)})
        start = stop
    return markers


def matched_pn(windows, reference):
    projection = (windows @ reference.conj()).real
    power = np.linalg.norm(windows, axis=-1) * np.linalg.norm(reference)
    return projection / np.maximum(power, 1e-30)


def map_pn(bits, points, labels):
    index = (bits[:, None, :] == labels[None]).all(axis=-1)
    if not (index.sum(axis=1) == 1).all():
        raise ValueError('PN label is ambiguous for this constellation')
    return points[index.argmax(axis=1)]


def audit_marker(values, marker, reference, constellation, orders, target):
    points, labels = constellation
    lanes = labels.shape[1]
    center = marker['full_systematic_stop']
    cell = marker['full_systematic_symbols']
    next_cell = cell + marker['systematic_mixed_skip']
    before = map_pn(reference[(cell - WINDOW) * lanes:cell * lanes].reshape(-1, lanes), points, labels)
    after = map_pn(reference[next_cell * lanes:(next_cell + WINDOW) * lanes].reshape(-1, lanes), points, labels)
    support = np.lib.stride_tricks.sliding_window_view(values[center - WINDOW - HALF_WIDTH:center + HALF_WIDTH], WINDOW)
    scores = matched_pn(support, before)
    wrong = before.reshape(-1, 16)[orders].reshape(len(orders), WINDOW)
    shuffled_scores = np.array([matched_pn(support[HALF_WIDTH], pattern) for pattern in wrong])
    continued_at = center + marker['systematic_mixed_skip']
    follow = matched_pn(values[continued_at:continued_at + WINDOW], after)
    selected = float(scores[HALF_WIDTH])
    return {'mcs': marker['mcs'], 'predicted_score': selected, 'continuation_score': float(follow),
            'peak_shift': int(scores.argmax()) - HALF_WIDTH, 'rank': int(np.count_nonzero(scores >= selected)),
            'strongest_other_shift': float(np.max(np.delete(scores, HALF_WIDTH))),
            'strongest_wrong_template': float(shuffled_scores.max()),
            'score_for_all_shifts': scores.tolist() if target else None,
            'shuffled_template_scores': shuffled_scores.tolist() if target else None}


def main():
    require_run_context()
    hashes = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(art(run, exp, name))

    a, b = [doc(f'v107-ut-offset-constrained-h2-{suffix}', 'E109', 'ut_offset_constrained_h2.json')
            for suffix in ('final', 'confirm')]
    if a != b:
        raise ValueError('Header-only run replicas differ')
    candidate, = [row for row in a['weak_frames'] if row['local_phase']['crc_ok']]
    header = candidate['local_phase']
    if candidate['other_pilot_fit_local_phase']['words_hex'] != header['words_hex']:
        raise ValueError('Alternate pilot timing produces a different H2 hypothesis')
    h2 = a['h2_start_resource']
    markers = predicted_markers(h2 + header['coding_length'], header['layout']['entries'])
    grid = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')
    sync = {row['tick']: row['sync'] for row in grid['offset_grid']['records'] if 'sync' in row}
    clock = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    capture = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    pn_model = doc('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json')['fit']['winner']
    reference = equality_predictions(np.arange(16384), pn_bits(), pn_model)
    with np.load(art('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as src:
        qam16 = src['constellation_points'], src['frozen_gray_mapping']
    mapping = doc('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')['cross32_mapping']
    qam32 = (np.array([complex(item['i'], item['q']) for item in mapping]) / np.sqrt(20),
             labels_as_bits(np.array([item['label'] for item in mapping]), 5))
    modes = {'16QAM': qam16, 'cross32': qam32}
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    sss_hash = sha256(sss_path)
    if sss_hash != capture['sss_reference_sha256']:
        raise ValueError('SSS fingerprint changed')
    hashes[PROJECT_ROOT / 'references/qin/sssVec.mat'] = sss_hash
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    raw_path = PROJECT_ROOT / SOURCE
    samples = load_samples(raw_path, capture)
    hashes[raw_path] = capture['source_sha256']
    rng = np.random.default_rng(SEED)
    orders = np.array([rng.permutation(WINDOW // 16) for _ in range(CONTROL_COUNT)])
    rows = []
    for fold, timing in enumerate(item['timing'] for item in clock['folds']):
        for tick in sorted(row['tick'] for row in a['weak_frames']):
            spectra = read_grid(samples, sync[tick], timing['sampling_scale'], timing['cfo_residual_hz'])
            channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(spectra[0] / sss, 2)])
            values = (spectra[1:, ~PILOT] * PN[:, ~PILOT].conj() * channel[~PILOT].conj()).ravel()
            if markers[-1]['word_stop'] > len(values):
                raise ValueError('Frozen allocations outside the available body aperture')
            oriented = np.concatenate((values[:h2], local_prefix_phase(values[h2:])))
            target = tick == candidate['tick']
            checks = [audit_marker(oriented, marker, reference, modes[KINDS[marker['mcs']]], orders, target)
                      for marker in markers]
            rows.append({'timing_fold': fold, 'tick': tick, 'header_candidate': target, 'checks': checks})
            if target:
                print('target', tick, 'fit', fold, [(r['mcs'], r['predicted_score'], r['rank']) for r in checks], flush=True)
    result = {'schema': 'starlink-ut-offset-h2-body/v1', 'seed': SEED, 'window_symbols': WINDOW,
              'search_half_width': HALF_WIDTH, 'shuffle_orders': orders.tolist(),
              'target_tick': candidate['tick'], 'h2_words_hex': header['words_hex'], 'markers': markers,
              'inputs_sha256': {path.as_posix(): digest for path, digest in hashes.items()}, 'rows': rows}
    out = experiment_output('E109')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ut_offset_h2_body.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
