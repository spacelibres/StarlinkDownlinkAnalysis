"""Test a published Code 37 shortening relation against frozen Code 21."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'learning')]
from project_paths import experiment_output, legacy_artifact, require_run_context
from allocation_core import predict_end
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, recover_generator, reference_gain
from code9_core import edge_slope_lifts, split_frames
from code37_core import LAST_BIT, PARITY_START, SYMBOLS, SYSTEMATIC, bits_from_labels, core_matrix, decide_word, pack_rows
from educational_fec_demo import encode_block, word_bits
from validate_code21 import HEADERS, SOURCE, digest

SEED = 2026112017
MOTHER = 'runs/v014-code21-repeat/artifacts/experiments/E016/results/code21_observations.npz'
GRID_INDICES = [0, 2, 5, 7, 8, 10, 13, 15]
LABELS = [1, 5, 4, 7, 0, 6, 2, 3]
RESOLVED = {'zero_initial_syndrome', 'resolved_in_search'}


def constellation_for(rotation, reflection, complement):
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    points = (levels[:, None] + 1j * levels[None, :]).ravel()[GRID_INDICES]
    if reflection:
        points = points.conj()
    table = np.empty(8, dtype=np.complex128)
    table[np.asarray(LABELS) ^ (7 if complement else 0)] = points * (1j ** rotation)
    return table


def receive(iq, frames, sss):
    prefix = word_bits(0x3048)[16:]
    signs = 1 - 2 * encode_block(np.r_[prefix, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    pilots = PILOTS[TRAIN_PILOTS]
    left, right = pilots[pilots < 0].mean(), pilots[pilots > 0].mean()
    data_carriers = LOADED[~np.isin(LOADED, PILOTS)]
    output, rows = [], []
    for row in frames:
        frame = row['legacy_frame_index']
        h2 = row['second_header']
        alloc = predict_end(h2['start_resource'], h2['blocks'], h2['layout20']['entries'])
        targets = [r for r in alloc['allocations'] if r['code'] == 37]
        if not targets:
            continue
        spectrum = fft_grid(iq, frame)
        equal = equalize_grid(spectrum[1:], spectrum[0], sss, np.arange(2, 302))
        phase, _ = edge_slope_lifts(equal['calibration'][:, 1], data_carriers, left, right)
        data = (equal['nonpilot'] * np.exp(-1j * phase)).ravel()
        data /= reference_gain(data[:48], signs)
        for field in targets:
            for word in range(field['quantity']):
                start = field['start_resource'] + word * SYMBOLS
                stop = start + SYMBOLS
                if not (0 < start < stop < len(data)):
                    raise ValueError('Code 37 control window unavailable')
                segment = data[start - 1:stop + 1]
                if len(segment) != SYMBOLS + 2:
                    raise ValueError('Incomplete Code 37 resource window')
                output.append(segment)
                rows.append({'frame': frame, 'word': word, 'start_resource': start})
    return np.asarray(output), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extended-search', action='store_true', help='Test up to four moves in the top 256')
    options = parser.parse_args()
    pair_limit = 256 if options.extended_search else 64
    max_changes = 4 if options.extended_search else 2
    require_run_context()
    out = experiment_output('E017')
    out.mkdir(parents=True, exist_ok=True)
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    inputs = {key: digest(ROOT / key) for key in [SOURCE, HEADERS, MOTHER]}
    expected = ['a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
                'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838',
                'd5b781862d4289aea615621eb95f06a75e10239f8bb100442428974a98bf9ac0']
    if list(inputs.values()) != expected:
        raise RuntimeError('Frozen inputs changed')
    with np.load(ROOT / MOTHER, allow_pickle=False) as prior:
        p = prior['parity_generator']
    matrix = core_matrix(p)
    columns = pack_rows(matrix)
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    h = json.loads((ROOT / HEADERS).read_text(encoding='utf-8'))
    observed, words = receive(np.load(ROOT / SOURCE, mmap_mode='r'), h['frames'], sss)
    samples = observed[:, 1:-1]
    frames = np.array([r['frame'] for r in words])
    train, order = split_frames(frames, SEED, train_groups=2)
    trials = []
    for rotation in range(4):
        for reflection in [False, True]:
            for complement in [False, True]:
                points = constellation_for(rotation, reflection, complement)
                tested = [decide_word(z, points, matrix, columns, pair_limit, max_changes)[2] for z in samples[train]]
                trials.append({'rotation_quarters': rotation, 'reflection': reflection, 'complement': complement,
                               'training_resolved': sum(r['status'] in RESOLVED for r in tested),
                               'train_word_results': tested})
    accepted = [r for r in trials if r['training_resolved'] == int(train.sum())]
    result = {'schema': 'starlink-code37-shortening/v1', 'input_sha256': inputs, 'sss_sha256': digest(sss_path),
              'mother_generator_sha256': hashlib.sha256(p.tobytes()).hexdigest(), 'coordinate_scope': {
                  'observed_8193_bits': [[0, SYSTEMATIC], [SYSTEMATIC, PARITY_START], [PARITY_START, LAST_BIT], [LAST_BIT, LAST_BIT + 1]],
                  'virtual_zero_mother_start': [0, 384]}, 'group_seed': SEED, 'group_order': order,
              'training_frames': np.unique(frames[train]).tolist(), 'reserved_frames': np.unique(frames[~train]).tolist(),
              'label_bank': trials, 'accepted_count': len(accepted), 'words': words, 'search_shortlist': pair_limit,
              'max_symbol_changes': max_changes}
    arrays = {'observed_symbols_with_neighbors': observed, 'training_frames': train}
    if len(accepted) == 1:
        selected = accepted[0]
        result['selected'] = {k: v for k, v in selected.items() if k != 'train_word_results'}
        points = constellation_for(selected['rotation_quarters'], selected['reflection'], selected['complement'])
        reports = [decide_word(z, points, matrix, columns, pair_limit, max_changes) for z in samples]
        hard, corrected, decisions = zip(*reports)
        hard, corrected = np.asarray(hard), np.asarray(corrected)
        core = np.r_[np.ones(SYSTEMATIC, bool), np.zeros(384, bool), np.ones(3072, bool), False]
        known_symbols = core.reshape(SYMBOLS, 3).all(axis=1)
        for i, row in enumerate(words):
            b = corrected[i]
            received_parity = b[PARITY_START:LAST_BIT]
            value = samples[i, known_symbols]
            labels = b.reshape(-1, 3)[known_symbols] @ np.array([4, 2, 1])
            parity_flip = b.copy()
            parity_flip[PARITY_START:LAST_BIT:16] ^= 1
            flip_labels = parity_flip.reshape(-1, 3)[known_symbols] @ np.array([4, 2, 1])
            row.update(decisions[i], training=bool(train[i]), final_syndrome_weight=int(((b @ matrix) % 2).sum()),
                       full_mother_observed_midsegment_errors=int(np.count_nonzero((b[:PARITY_START] @ p) % 2 != received_parity)),
                       virtual_zero_at_end_errors=int(np.count_nonzero((b[:SYSTEMATIC] @ p[:SYSTEMATIC]) % 2 != received_parity)),
                       unknown_midsegment_ones=int(b[SYSTEMATIC:PARITY_START].sum()),
                       observed_spare_label_bit=int(b[LAST_BIT]),
                       core_symbol_nmse=float(np.sum(abs(value - points[labels]) ** 2) / np.sum(abs(value) ** 2)),
                       fixed_parity_flip_core_symbol_nmse=float(np.sum(abs(value - points[flip_labels]) ** 2) / np.sum(abs(value) ** 2)))
            for shift in [-1, 1]:
                start = 1 + shift
                labels_shifted = np.argmin(abs(observed[i, start:start + SYMBOLS, None] - points) ** 2, axis=1).astype(np.uint8)
                row[f'shift_{shift}_hard_syndrome_weight'] = int(((bits_from_labels(labels_shifted) @ matrix) % 2).sum())
        arrays.update(hard_bits=hard, corrected_bits=corrected, core_coordinate_mask=core, constellation=points)
        result['fully_core_symbols_per_word'] = int(known_symbols.sum())
        result['reserved_verified_words'] = sum(not w['training'] and w['final_syndrome_weight'] == 0 for w in words)
        result['reserved_total_words'] = int((~train).sum())
        unknown = hard[:, SYSTEMATIC:PARITY_START]
        pair_distances = np.count_nonzero(unknown[:, None] != unknown[None, :], axis=2)
        result['unexplained_hard_observations'] = {
            'cross_word_constant_positions': int(np.count_nonzero(np.all(unknown == unknown[0], axis=0))),
            'pairwise_hamming_distances_upper_triangle': pair_distances[np.triu_indices(len(unknown), 1)].tolist(),
            'preserved_after_correction': bool(np.array_equal(hard[:, ~core], corrected[:, ~core])),
        }
        raw_residual = ((hard[:, :SYSTEMATIC] @ p[384:]) % 2) ^ hard[:, PARITY_START:LAST_BIT]
        result['raw_prefix_completion'] = []
        for residual in raw_residual:
            completion = recover_generator(p[:384].T, residual[:, None])
            coordinate_values = completion.pop('generator')
            completion['ones_in_unique_completion'] = None if coordinate_values is None else int(coordinate_values.sum())
            result['raw_prefix_completion'].append(completion)
    (out / 'code37.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'code37_observations.npz', **arrays)
    print('Accepted training conventions', len(accepted),
          'reserved', result.get('reserved_verified_words'), '/', result.get('reserved_total_words'), flush=True)
    if len(accepted) != 1:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
