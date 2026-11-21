"""Test fixed code relations and their controls on the legacy capture."""
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
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, reference_gain
from code9_core import edge_slope_lifts
from code21_ring import binary_blocks, evaluate, pack_blocks
from educational_fec_demo import encode_block, word_bits
from qc_figure_core import dense_checks, generator_from_shifts, one_defect_span, qc_syndrome, symbol_llrs
from sparse_code5_core import normalized_min_sum
from validate_code21 import HEADERS, SOURCE, digest
from validate_code37 import constellation_for

TABLE = 'experiments/E018/figure12_shifts.json'
OLD_H = 'runs/v011-sparse-code5/artifacts/experiments/E013/results/sparse_code5_arrays.npz'
CONFIG = {3: (512, 8192), 12: (512, 4096), 27: (512, 2731), 10: (1024, 8192), 39: (1024, 4096)}


def constellation(code):
    if code == 3:
        return np.array([1., -1.]), np.array([[0], [1]], dtype=np.uint8)
    if code in (10, 12):
        points = np.exp(1j * (np.pi / 4 + np.arange(4) * np.pi / 2))
        labels = np.array([3, 1, 0, 2], dtype=np.uint8)
        return points, np.column_stack((labels >> 1, labels & 1))
    if code == 27:
        points = constellation_for(2, False, False)
        return points, ((np.arange(8)[:, None] >> np.array([2, 1, 0])) & 1).astype(np.uint8)
    if code == 39:
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        i, q = [a.ravel() for a in np.meshgrid(np.arange(4), np.arange(4), indexing='ij')]
        gray = np.array([0, 1, 3, 2], dtype=np.uint8)
        i_bits, q_bits = gray[i], gray[q]
        return -(levels[i] + 1j * levels[q]), np.column_stack((i_bits >> 1, q_bits >> 1, i_bits & 1, q_bits & 1))
    raise ValueError('Unknown modulation')


def extract(iq, frames, sss):
    pilots = PILOTS[TRAIN_PILOTS]
    left, right = pilots[pilots < 0].mean(), pilots[pilots > 0].mean()
    carriers = LOADED[~np.isin(LOADED, PILOTS)]
    prefix = word_bits(0x3048)[16:]
    signs = 1 - 2 * encode_block(np.r_[prefix, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    for record in frames:
        f = record['legacy_frame_index']
        h = record['second_header']
        regions = predict_end(h['start_resource'], h['blocks'], h['layout20']['entries'])['allocations']
        targets = [r for r in regions if r['code'] in CONFIG]
        if not targets:
            continue
        grid = fft_grid(iq, f)
        eq = equalize_grid(grid[1:], grid[0], sss, np.arange(2, 302))
        phase, _ = edge_slope_lifts(eq['calibration'][:, 1], carriers, left, right)
        flat = (eq['nonpilot'] * np.exp(-1j * phase)).ravel()
        flat /= reference_gain(flat[:48], signs)
        for r in targets:
            code = r['code']
            lift, length = CONFIG[code]
            for index in range(r['quantity']):
                start = r['start_resource'] + index * length
                stop = start + length
                if not 0 < start < stop < len(flat):
                    raise ValueError('Required raw or control window is incomplete')
                yield f, code, index, start, lift, flat[start - 1:stop + 1]


def block_parity_constraints(words, lift):
    block_parity = words.reshape(len(words), -1, lift).sum(axis=-1) % 2
    masks = (np.arange(1, 256)[:, None] >> np.arange(8)) & 1
    return [np.flatnonzero(mask).tolist() for mask in masks if not ((block_parity @ mask) % 2).any()]


def main():
    require_run_context()
    out = experiment_output('E018')
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = {path: digest(ROOT / path) for path in [SOURCE, HEADERS, TABLE, OLD_H]}
    expected = ['a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
                'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838']
    if list(input_hashes.values())[:2] != expected:
        raise RuntimeError('Frozen waveform or headers changed')
    table = json.loads((ROOT / TABLE).read_text(encoding='utf-8'))
    shifts = {z: np.asarray(table[str(z)], dtype=int) for z in [128, 512, 1024]}
    with np.load(ROOT / OLD_H, allow_pickle=False) as known:
        old_h = known['check_matrix']
    published_h = dense_checks(shifts[128], 128)
    if set(map(bytes, old_h)) != set(map(bytes, published_h)):
        raise RuntimeError('Transcribed Code 5 checks differ in row support')
    p = {z: generator_from_shifts(s, z) for z, s in shifts.items()}
    arrays = {f'figure_{z}_shifts': s for z, s in shifts.items()}
    result = {'schema': 'starlink-qc-figure/v1', 'input_sha256': input_hashes,
              'source_figure': table['source'], 'code5_same_check_row_set': True,
              'shared_incidence': all(np.array_equal(shifts[128] >= 0, s >= 0) for s in shifts.values()),
              'check_rank_from_invertible_parity_side': {z: 8 * z for z in shifts},
              'generators_hex': {z: [[f'{b:0{z//4}x}' for b in row] for row in v] for z, v in p.items()},
              'word_results': []}
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    result['sss_sha256'] = digest(sss_path)
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    headers = json.loads((ROOT / HEADERS).read_text(encoding='utf-8'))['frames']
    iq = np.load(ROOT / SOURCE, mmap_mode='r', allow_pickle=False)
    code10 = []
    code39 = []
    for f, code, index, start, lift, windows in extract(iq, headers, sss):
        obs = windows[1:-1]
        label, mapping = constellation(code)
        k, n = 8 * lift, 16 * lift
        llr = symbol_llrs(obs, label, mapping)
        bits = (llr[:n] < 0).astype(np.uint8)
        h0 = qc_syndrome(bits[None, :], shifts[lift], lift)
        if h0.any():
            h = dense_checks(shifts[lift], lift)
            decoded = normalized_min_sum(llr[:n], h, eta=.8, max_iterations=60)
            corrected = decoded.pop('bits')
            del h
        else:
            corrected = bits.copy()
            decoded = {'converged': True, 'iterations': 0, 'initial_failed_checks': 0, 'final_failed_checks': 0}
        final = qc_syndrome(corrected[None, :], shifts[lift], lift)
        parity = binary_blocks(evaluate(pack_blocks(corrected[None, :k], lift), p[lift], lift), lift)[0]
        if not np.array_equal(final.any(), np.any(parity != corrected[k:])):
            raise RuntimeError('QC check and independent generator disagree')
        width = mapping.shape[1]
        used = n // width
        word_labels = corrected[:used * width].reshape(-1, width)
        mapped = np.all(word_labels[:, None, :] == mapping[None, :, :], axis=-1).argmax(axis=1)
        nmse = np.sum(abs(obs[:used] - label[mapped]) ** 2) / np.sum(abs(obs[:used]) ** 2)
        negatives = {}
        for shift in [-1, 1]:
            x = windows[1 + shift:1 + shift + len(obs)]
            wrong = symbol_llrs(x, label, mapping)[:n] < 0
            negatives[str(shift)] = int(qc_syndrome(wrong[None, :], shifts[lift], lift).sum())
        alternatives = {}
        for name, ratio in [('zero_embed', 1), ('scaled', lift // 128)]:
            wrong = np.where(shifts[128] >= 0, shifts[128] * ratio, -1)
            alternatives[name] = int(qc_syndrome(corrected[None, :], wrong, lift).sum())
        row = {'frame': f, 'code': code, 'word': index, 'start_resource': start, **decoded,
               'changed_positions': np.flatnonzero(corrected != bits).tolist(),
               'parity_reencoding_disagreements': int(np.count_nonzero(parity != corrected[k:])),
               'wrong_start_failed_checks': negatives, 'code5_shift_transfer_failed_checks': alternatives,
               'complement_failed_checks': int(qc_syndrome((1-corrected)[None, :], shifts[lift], lift).sum()),
               'fully_coded_symbol_count': used, 'full_word_remodulation_nmse_over_complete_labels': float(nmse),
               'unassigned_label_hard_bits': (llr[n:] < 0).astype(int).tolist()}
        result['word_results'].append(row)
        key = f'frame_{f}_code_{code}_word_{index}'
        arrays[f'{key}_symbols'] = obs
        arrays[f'{key}_hard_bits'] = bits
        arrays[f'{key}_decoded_bits'] = corrected
        if code == 10 and not final.any():
            code10.append(corrected[:k])
        if code == 39 and not final.any():
            code39.append((bits[:k], corrected[:k]))
        print(f, code, index, int(h0.sum()), '->', int(final.sum()), flush=True)
    result['code10_block_parity_constraints_zero_based'] = block_parity_constraints(np.asarray(code10), 1024)
    result['code10_qc_input_span'] = one_defect_span(pack_blocks(np.asarray(code10), 1024), 1024)
    result['code10_code39_decoded_qc_input_span'] = one_defect_span(
        pack_blocks(np.asarray(code10 + [decoded for _, decoded in code39]), 1024), 1024)
    result['code39_hard_and_decoded_constraint_violations'] = [
        [[int(word.reshape(8, 1024)[mask].sum() % 2)
          for mask in result['code10_block_parity_constraints_zero_based']] for word in pair]
        for pair in code39
    ]
    result['observed_words'] = len(result['word_results'])
    result['validated_words'] = sum(r['parity_reencoding_disagreements'] == 0 for r in result['word_results'])
    (out / 'qc_figure.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'qc_figure_observations.npz', **arrays)


if __name__ == '__main__':
    main()
