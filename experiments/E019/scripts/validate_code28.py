"""Recover the Code 28 mother generator and test its shortened relation."""
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
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, reference_gain, recover_generator
from code9_core import edge_slope_lifts, split_frames
from code21_ring import binary_blocks, evaluate, pack_blocks, recover
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, SYSTEMATIC, coordinates, packed_check_columns, repair_one
from educational_fec_demo import encode_block, word_bits
from validate_code21 import HEADERS, SOURCE, digest
from validate_code37 import constellation_for

SEED = 2026112019
FIT_COUNT = 20
SHORTLIST = 256


def receive(iq, frames, sss):
    known = word_bits(0x3048)[16:]
    signs = 1 - 2 * encode_block(np.r_[known, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    train_pilots = PILOTS[TRAIN_PILOTS]
    left, right = train_pilots[train_pilots < 0].mean(), train_pilots[train_pilots > 0].mean()
    data = LOADED[~np.isin(LOADED, PILOTS)]
    observed, rows = [], []
    for frame in frames:
        f = frame['legacy_frame_index']
        h = frame['second_header']
        alloc = predict_end(h['start_resource'], h['blocks'], h['layout20']['entries'])
        regions = [r for r in alloc['allocations'] if r['code'] == 28]
        if not regions:
            continue
        grid = fft_grid(iq, f)
        eq = equalize_grid(grid[1:], grid[0], sss, np.arange(2, 302))
        phase, _ = edge_slope_lifts(eq['calibration'][:, 1], data, left, right)
        samples = (eq['nonpilot'] * np.exp(-1j * phase)).ravel()
        samples /= reference_gain(samples[:48], signs)
        for region in regions:
            for i in range(region['quantity']):
                start = region['start_resource'] + i * SYMBOLS
                stop = start + SYMBOLS
                if not 0 < start < stop < len(samples):
                    raise ValueError('Incomplete observation or control')
                observed.append(samples[start - 1:stop + 1].astype(np.complex64))
                rows.append({'frame': f, 'word': i, 'start_resource': start})
    return np.asarray(observed), rows


def labels_to_bits(labels):
    return ((labels[..., None].astype(np.uint8) >> np.array([2, 1, 0], np.uint8)) & 1).reshape(len(labels), -1)


def predict(u, matrix):
    return binary_blocks(evaluate(pack_blocks(u, LIFT), matrix, LIFT), LIFT)


def fit(u, v, scores, group, frames):
    candidates = np.flatnonzero(group)
    indices = candidates[np.argsort(-scores[candidates], kind='stable')[:FIT_COUNT]]
    p, rank, inconsistencies = recover(pack_blocks(u[indices], LIFT), pack_blocks(v[indices], LIFT), LIFT)
    return p, {'fit_indices': indices.tolist(), 'fit_frames': np.unique(frames[group]).tolist(),
               'reserved_frames': np.unique(frames[~group]).tolist(), 'unit_pivots': rank,
               'inconsistent_block_equations': inconsistencies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', action='store_true', help='Run bounded four-symbol search on nonzero syndromes')
    args = parser.parse_args()
    require_run_context()
    out = experiment_output('E019')
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = {p: digest(ROOT / p) for p in [SOURCE, HEADERS]}
    if list(input_hashes.values()) != ['a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
                                       'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838']:
        raise RuntimeError('Frozen inputs changed')
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    input_hashes['references/qin/sssVec.mat'] = digest(sss_path)
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    headers = json.loads((ROOT / HEADERS).read_text(encoding='utf-8'))['frames']
    window, rows = receive(np.load(ROOT / SOURCE, mmap_mode='r', allow_pickle=False), headers, sss)
    samples = window[:, 1:-1]
    frame_ids = np.array([r['frame'] for r in rows])
    constellation = constellation_for(2, False, True)
    distances = abs(samples[..., None] - constellation) ** 2
    labels = distances.argmin(axis=-1)
    closest = np.partition(distances, 1, axis=-1)[..., :2]
    margin = (closest[..., 1] - closest[..., 0]).min(axis=1)
    hard = labels_to_bits(labels)
    u, v, mask = coordinates(hard)
    published = np.isin(frame_ids, [1, 5])
    random, order = split_frames(frame_ids, SEED, 2)
    p, pub = fit(u, v, margin, published, frame_ids)
    q, rand = fit(u, v, margin, random, frame_ids)
    arrays = {'observations_with_neighbors': window, 'hard_bits': hard, 'core_coordinate_mask': mask,
              'minimum_decision_margin': margin, 'published_group': published, 'random_group': random}
    result = {'schema': 'starlink-code28-shortening/v1', 'input_sha256': input_hashes,
              'words': rows, 'published_fit': pub, 'randomized_fit': rand, 'random_seed': SEED,
              'group_order': order, 'matrix_agreement': p is not None and p == q,
              'receiver_mapping': [2, False, True], 'repair_enabled': args.repair}
    for name, fitted in [('published', pub), ('randomized', rand)]:
        print(name, fitted['unit_pivots'], fitted['inconsistent_block_equations'], flush=True)
    if result['matrix_agreement']:
        raw_syndromes = predict(u, p) ^ v
        result['polynomials_hex'] = [[f'{b:0256x}' for b in row] for row in p]
        p_bits = binary_blocks(p, LIFT)
        result['cyclic_seed_sha256'] = hashlib.sha256(p_bits.tobytes()).hexdigest()
        corrected = hard.copy()
        columns = packed_check_columns(p, LIFT, OMITTED, 2) if args.repair else None
        for j, row in enumerate(rows):
            syndrome = raw_syndromes[j]
            changes, record = (repair_one(labels[j], distances[j], syndrome, columns, mask, SHORTLIST)
                               if args.repair else ([], {'status': 'hard_only'}))
            corrected[j, changes] ^= 1
            row.update(record, initial_failed_checks=int(syndrome.sum()), changed_positions=changes)
        final_u, final_v, _ = coordinates(corrected)
        end = predict(final_u, p) ^ final_v
        for i, row in enumerate(rows):
            row['final_failed_checks'] = int(end[i].sum())
            if row['status'] == 'resolved_in_search' and row['final_failed_checks']:
                raise RuntimeError('Syndrome search disagrees with polynomial encoder')
        for group, fitted in [(published, pub), (random, rand)]:
            fitted.update(reserved_words=int((~group).sum()),
                          reserved_hard_exact_words=int(np.count_nonzero(~group & ~raw_syndromes.any(axis=1))),
                          reserved_final_exact_words=int(np.count_nonzero(~group & ~end.any(axis=1))))
        for name, word in [('full_observed_middle', corrected[:, :MOTHER_K]),
                           ('virtual_zero_at_end', np.column_stack((corrected[:, :SYSTEMATIC], np.zeros((len(hard), OMITTED), np.uint8))))]:
            result[name + '_errors'] = np.count_nonzero(predict(word, p) != final_v, axis=1).tolist()
        cu, cv, _ = coordinates(1 - corrected)
        result['complement_errors'] = np.count_nonzero(predict(cu, p) != cv, axis=1).tolist()
        result['resource_shift_errors'] = {}
        for shift in [-1, 1]:
            first = 1 + shift
            test = labels_to_bits((abs(window[:, first:first + SYMBOLS, None] - constellation)**2).argmin(axis=-1))
            tu, tv, _ = coordinates(test)
            result['resource_shift_errors'][str(shift)] = np.count_nonzero(predict(tu, p) != tv, axis=1).tolist()
        if args.repair:
            # Earliest raw-valid word, irrespective of its final correction outcome.
            clean = np.flatnonzero(~raw_syndromes.any(axis=1))
            if len(clean):
                prefix = packed_check_columns(p[:1], LIFT, 0, 0)[:OMITTED]
                width = (MOTHER_N - MOTHER_K) // 8
                cols = np.frombuffer(b''.join(v.to_bytes(width, 'little') for v in prefix),
                                     dtype=np.uint8).reshape(OMITTED, width)
                matrix = np.unpackbits(cols, axis=1, bitorder='little').T
                recovered = recover_generator(matrix, raw_syndromes[clean[0], :, None])
                g = recovered.pop('generator')
                result['raw_prefix_completion'] = {'word_index': int(clean[0]), **recovered,
                                                   'ones': None if g is None else int(g.sum())}
        arrays.update(corrected_bits=corrected, generator_seed_bits=p_bits, initial_syndromes=raw_syndromes,
                      final_syndromes=end, constellation=constellation)
        result['verified_words'] = int(np.count_nonzero(~end.any(axis=1)))
        result['initially_verified_words'] = int(np.count_nonzero(~raw_syndromes.any(axis=1)))
        print('Verified', result['initially_verified_words'], '->', result['verified_words'], '/', len(rows), flush=True)
    (out / 'code28.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'code28_observations.npz', **arrays)
    if not result['matrix_agreement']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
