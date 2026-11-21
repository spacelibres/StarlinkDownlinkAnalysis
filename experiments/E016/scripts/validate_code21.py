"""Recover and validate the Code 21 QC generator on grouped observations."""
import argparse
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'learning')]
from project_paths import experiment_output, legacy_artifact, require_run_context
from allocation_core import predict_end
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, reference_gain
from code9_core import edge_slope_lifts, split_frames
from code21_ring import binary_blocks, dense_generator, evaluate, pack_blocks, recover, reduce_polynomial
from educational_fec_demo import encode_block, word_bits

LIFT, K, N = 512, 5120, 8192
SEED = 2026112115
PHASE_ZERO = np.pi / 4
LABELS = np.array([3, 1, 0, 2], dtype=np.uint8)
SOURCE = 'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
HEADERS = 'runs/v008-complete-header-layout-02/artifacts/experiments/E010/results/full_header_layout.json'
CODE9 = 'runs/v013-code9-time-unwrapped/artifacts/experiments/E015/results/code9_observations.npz'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def coordinates(values, orientation=1, quarter_shift=0):
    quarter = np.rint((np.angle(values) - PHASE_ZERO) / (np.pi / 2)).astype(int) % 4
    labels = LABELS[(orientation * quarter + quarter_shift) % 4]
    return np.stack((labels >> 1, labels & 1), axis=-1).reshape(len(values), -1)


def receive(iq, frames, sss):
    known = word_bits(0x3048)[16:]
    signs = 1 - 2 * encode_block(np.r_[known, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    carriers = PILOTS[TRAIN_PILOTS]
    left, right = carriers[carriers < 0].mean(), carriers[carriers > 0].mean()
    data_mask = ~np.isin(LOADED, PILOTS)
    samples, meta = [], []
    for frame in frames:
        index = frame['legacy_frame_index']
        h2 = frame['second_header']
        alloc = predict_end(h2['start_resource'], h2['blocks'], h2['layout20']['entries'])
        targets = [r for r in alloc['allocations'] if r['code'] == 21]
        if not targets:
            continue
        grid = fft_grid(iq, index)
        eq = equalize_grid(grid[1:], grid[0], sss, np.arange(2, 302))
        phase, _ = edge_slope_lifts(eq['calibration'][:, 1], LOADED[data_mask], left, right)
        grid = eq['nonpilot'] * np.exp(-1j * phase)
        resources = grid.ravel() / reference_gain(grid.ravel()[:48], signs)
        for target in targets:
            for word in range(target['quantity']):
                start = target['start_resource'] + word * (N // 2)
                stop = start + N // 2
                if start < 1 or stop >= len(resources):
                    raise ValueError('Target or adjacent control not fully observed')
                samples.append(resources[start - 1:stop + 1].astype(np.complex64))
                meta.append({'frame': index, 'word': word, 'start_resource': start})
    return np.asarray(samples), meta


def errors(inputs, observed, generator):
    predicted = binary_blocks(evaluate(inputs, generator, LIFT), LIFT)
    return np.count_nonzero(predicted != observed, axis=1)


def fit_convention(symbols, selected, search):
    candidates = [(d, q) for d in [1, -1] for q in range(4)] if search else [(1, 0)]
    records, valid = [], []
    for direction, shift in candidates:
        local = coordinates(symbols[selected], direction, shift)
        model, pivots, inconsistent = recover(pack_blocks(local[:, :K], LIFT), pack_blocks(local[:, K:], LIFT), LIFT)
        records.append({'orientation': direction, 'quarter_shift': shift,
                        'unit_pivots': pivots, 'inconsistent_block_equations': inconsistent})
        if model is not None:
            valid.append((model, direction, shift))
    if len(valid) != 1:
        return None, None, records
    model, direction, shift = valid[0]
    return model, (direction, shift), records


def check_assignment(symbols, frames, group, selected, search):
    model, convention, candidates = fit_convention(symbols, selected, search)
    record = {'fit_word_indices': selected.tolist(), 'fit_frames': np.unique(frames[group]).tolist(),
              'reserved_frames': np.unique(frames[~group]).tolist(), 'reserved_words': int((~group).sum()),
              'training_conventions': candidates, 'selected_convention': convention}
    if model is not None:
        bits = coordinates(symbols, *convention)
        differences = errors(pack_blocks(bits[:, :K], LIFT), bits[:, K:], model)
        record.update(parity_disagreements=differences.tolist(), calibration_frame_differences=int(differences[group].sum()),
                      reserved_parity_bits=int((~group).sum()) * (N - K),
                      reserved_parity_disagreements=int(differences[~group].sum()),
                      reserved_exact_words=int(np.count_nonzero(differences[~group] == 0)))
    return model, convention, record


def expand_exponents(value):
    return sum(1 << (4 * i) for i in range(128) if (value >> i) & 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-convention-bank', action='store_true')
    options = parser.parse_args()
    require_run_context()
    output = experiment_output('E016')
    output.mkdir(parents=True, exist_ok=True)
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    inputs = {p: digest(ROOT / p) for p in [SOURCE, HEADERS, CODE9]}
    expected = ['a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
                'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838',
                '2c8b6a82004e6787388d58cb53c0da82f82e2efab24d64975a90e055d1452292']
    if list(inputs.values()) != expected:
        raise RuntimeError('Frozen input changed')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    header_rows = json.loads((ROOT / HEADERS).read_text(encoding='utf-8'))['frames']
    iq = np.load(ROOT / SOURCE, mmap_mode='r', allow_pickle=False)
    observed, meta = receive(iq, header_rows, sss)
    symbols = observed[:, 1:-1]
    frames = np.array([w['frame'] for w in meta])
    published = np.isin(frames, [-1, 3])
    published_indices = np.flatnonzero(published)[:20]
    randomized, group_order = split_frames(frames, SEED, train_groups=2)
    min_margin = np.minimum(abs(symbols.real), abs(symbols.imag)).min(axis=1)
    eligible = np.flatnonzero(randomized)
    randomized_indices = eligible[np.argsort(-min_margin[eligible], kind='stable')[:20]]
    search = options.training_convention_bank
    model, convention, pub = check_assignment(symbols, frames, published, published_indices, search)
    model_r, conv_r, rand = check_assignment(symbols, frames, randomized, randomized_indices, search)
    bits = coordinates(symbols, *(convention or (1, 0)))
    packed_u = pack_blocks(bits[:, :K], LIFT)
    result = {'schema': 'starlink-code21-qc/v1', 'input_sha256': inputs, 'sss_sha256': digest(sss_path),
              'parameters': {'n': N, 'k': K, 'lift': LIFT}, 'quadrant_labels': LABELS.tolist(),
              'phase_zero_radians': PHASE_ZERO, 'group_seed': SEED, 'group_order': group_order,
              'words': meta, 'published_group': pub, 'randomized_group': rand,
              'equal_generators': model is not None and model == model_r and convention == conv_r}
    arrays = {'observations_with_edge_controls': observed, 'hard_bits': bits, 'published_training_frames': published,
              'randomized_training_frames': randomized, 'minimum_axis_margin': min_margin}
    for label, record in [('published', pub), ('randomized', rand)]:
        print(label, record['selected_convention'],
              record.get('reserved_parity_disagreements'), flush=True)
    if model is not None:
        p = dense_generator(model, LIFT)
        predicted = binary_blocks(evaluate(packed_u, model, LIFT), LIFT)
        if not np.array_equal((bits[::31, :K] @ p) % 2, predicted[::31]):
            raise RuntimeError('Ring and binary parity predictions differ')
        phase_by_label = np.empty(4)
        quarters = np.arange(4)
        phase_by_label[LABELS[(convention[0] * quarters + convention[1]) % 4]] = PHASE_ZERO + quarters * np.pi / 2
        for name, parity in [('prediction', predicted), ('fixed_flips', predicted.copy())]:
            if name == 'fixed_flips':
                parity[:, ::16] ^= 1
            labels = 2 * parity[:, ::2] + parity[:, 1::2]
            x = symbols[~published, K // 2:]
            s = np.exp(1j * phase_by_label[labels[~published]])
            result[name + '_reserved_nmse'] = float(np.sum(abs(x - s) ** 2) / np.sum(abs(x) ** 2))
        with np.load(ROOT / CODE9, allow_pickle=False) as prior:
            q9 = pack_blocks(prior['parity_generator'][::128], 128)
        transferred = [[expand_exponents(v) for v in row] for row in q9]
        reduced = [[reduce_polynomial(v, 128) for v in row] for row in model]
        result['code9_zero_padded_errors'] = errors(packed_u, bits[:, K:], q9).tolist()
        result['code9_exponent_times_four_errors'] = errors(packed_u, bits[:, K:], transferred).tolist()
        result['matching_polynomials_after_reduction'] = sum(a == b for ra, rb in zip(q9, reduced) for a, b in zip(ra, rb))
        controls = {'shift_minus_one': coordinates(observed[:, :-2], *convention), 'shift_plus_one': coordinates(observed[:, 2:], *convention),
                    'whole_complement': 1 - bits}
        result['controls'] = {key: errors(pack_blocks(w[:, :K], LIFT), w[:, K:], model).tolist() for key, w in controls.items()}
        result['expanded_generator_sha256'] = hashlib.sha256(p.tobytes()).hexdigest()
        result['polynomials_hex'] = [[f'{v:0128x}' for v in row] for row in model]
        arrays.update(parity_generator=p, predicted_parity=predicted)
    if model_r is not None:
        result['randomized_polynomials_hex'] = [[f'{v:0128x}' for v in row] for row in model_r]
    (output / 'code21.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(output / 'code21_observations.npz', **arrays)
    if model is None or model_r is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
