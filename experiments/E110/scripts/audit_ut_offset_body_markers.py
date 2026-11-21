"""Find coherent padding boundaries before reading variable H2 symbols."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS, WINDOW
from audit_internal_allocations import KINDS, WIDTH
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_offset_constrained_h2 import local_prefix_phase
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import capture_grid
from verify_ut_offset_h2_body import map_pn, matched_pn


SEED = 20261121111
BODY_START_GUARD = 7 * 114
BEFORE_FLOOR = .30
AFTER_LENGTH = 64
AFTER_MAX = .25
DROP_MIN = .16
CONTROL_COUNT = 31


def normalized_sliding_match(data, pattern):
    cumulative = np.r_[0., np.cumsum(abs(data)**2, dtype=float)]
    power = cumulative[len(pattern):] - cumulative[:-len(pattern)]
    projection = correlate(data, pattern, mode='valid', method='fft').real
    return np.divide(projection, np.linalg.norm(pattern) * np.sqrt(np.maximum(power, 1e-30)),
                     out=np.zeros_like(projection), where=power > 0)


def body_paths(markers, h2_start):
    def extend(start, marks, capacity, chain):
        for index, (code, stop) in enumerate(marks):
            count, slack = divmod(stop - start, WIDTH[code])
            if count <= 0 or slack or count >= 4096:
                continue
            next_chain = chain + ((code, count),)
            yield next_chain
            if len(next_chain) < capacity:
                yield from extend(stop, marks[index + 1:], capacity, next_chain)

    unique = sorted(set(markers), key=lambda pair: (pair[1], pair[0]))
    return [(blocks, path) for blocks in range(2, 8)
            for path in extend(h2_start + 114 * blocks, unique, min(15, (32 * blocks - 24) // 20), ())]


def frozen_templates(reference, modes, orders):
    result = {}
    for code, end in SYSTEMATIC_ENDS.items():
        points, labels = modes[KINDS[code]]
        lanes = labels.shape[1]
        cell, remainder = divmod(end, lanes)
        following = cell + int(remainder != 0)
        before = map_pn(reference[(cell - WINDOW) * lanes:cell * lanes].reshape(-1, lanes), points, labels)
        after = map_pn(reference[following * lanes:(following + AFTER_LENGTH) * lanes].reshape(-1, lanes), points, labels)
        shuffled = before.reshape(-1, 16)[orders].reshape(len(orders), WINDOW)
        result[code] = (cell, int(remainder != 0), before, after, shuffled)
    return result


def search_markers(data, origin, templates):
    hits = []
    for code, (cell, mixed, before, after, shuffles) in templates.items():
        scores = normalized_sliding_match(data, before)
        for begin in np.flatnonzero(scores >= BEFORE_FLOOR):
            stop = int(begin) + WINDOW
            word_stop = stop + WIDTH[code] - cell
            after_start = stop + mixed
            if word_stop > len(data) or after_start + AFTER_LENGTH > len(data):
                continue
            continuation = float(matched_pn(data[after_start:after_start + AFTER_LENGTH], after))
            score = float(scores[begin])
            control = max(float(matched_pn(data[begin:stop], s)) for s in shuffles)
            hits.append({'mcs': code, 'systematic_stop': origin + stop, 'word_stop': origin + word_stop,
                         'before': score, 'continued_pn': continuation, 'max_template_control': control,
                         'passed': continuation <= AFTER_MAX and score - continuation >= DROP_MIN and score > control})
    return hits


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    layout = doc('v043-ut-offset-h2-final', 'E045', 'ut_offset_h2_systematics.json')
    h2 = layout['h2_start_resource']
    input_ticks = {tick: group for group, values in layout['tick_groups'].items() for tick in values}
    epochs = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')['offset_grid']['records']
    sync = {row['tick']: row['sync'] for row in epochs if row['tick'] in input_ticks}
    clocks = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')['folds']
    capture = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    model = doc('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json')['fit']['winner']
    reference = equality_predictions(np.arange(16384), pn_bits(), model)
    with np.load(artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as src:
        modes = {'16QAM': (src['constellation_points'], src['frozen_gray_mapping'])}
    mapping = doc('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')['cross32_mapping']
    modes['cross32'] = (np.array([complex(item['i'], item['q']) for item in mapping]) / np.sqrt(20),
                        labels_as_bits(np.array([item['label'] for item in mapping]), 5))
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    sss_hash = sha256(sss_path)
    if sss_hash != capture['sss_reference_sha256']:
        raise ValueError('Unexpected SSS fingerprint')
    hashes[PROJECT_ROOT / 'references/qin/sssVec.mat'] = sss_hash
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    path = PROJECT_ROOT / SOURCE
    samples = load_samples(path, capture)
    hashes[path] = capture['source_sha256']
    rotors = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in LOADED[~PILOT]])
    rng = np.random.default_rng(SEED)
    orders = np.array([rng.permutation(WINDOW // 16) for _ in range(CONTROL_COUNT)])
    templates = frozen_templates(reference, modes, orders)
    rows = []
    for tick, group in sorted(input_ticks.items()):
        fits = []
        for fold in clocks:
            fit = fold['timing']
            anchor = sync[tick]
            grid, slots = capture_grid(samples, anchor['epoch_480_samples'], fit['sampling_scale'],
                                       anchor['cfo_hz'] + fit['cfo_residual_hz'])
            channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(grid[0] / sss, 2)])
            soft = (grid[1:, ~PILOT] * rotors[:len(slots) - 1].conj() * channel[~PILOT].conj()).ravel()
            raw = local_prefix_phase(soft[h2:])[BODY_START_GUARD:]
            hits = search_markers(raw, h2 + BODY_START_GUARD, templates)
            fits.append({'resources': len(soft), 'hits': hits})
        support = [{(r['mcs'], r['word_stop']) for r in item['hits'] if r['passed']} for item in fits]
        common = sorted(support[0] & support[1], key=lambda pair: (pair[1], pair[0]))
        paths = body_paths(common, h2)
        rows.append({'tick': tick, 'group': group, 'fits': fits, 'common_markers': common, 'body_paths': paths})
        print(tick, len(common), len(paths), flush=True)
    result = {'schema': 'starlink-ut-offset-body-markers/v1', 'seed': SEED, 'h2_start': h2,
              'body_search_start': h2 + BODY_START_GUARD, 'inputs_sha256': {p.as_posix(): d for p, d in hashes.items()},
              'thresholds': [BEFORE_FLOOR, AFTER_MAX, DROP_MIN], 'shuffle_orders': orders.tolist(), 'rows': rows}
    out = experiment_output('E110')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ut_offset_body_markers.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
