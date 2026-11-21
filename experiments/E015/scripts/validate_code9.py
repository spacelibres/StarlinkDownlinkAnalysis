"""Recover and test Code 9 using two declared whole-frame assignments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'project.json').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'learning')]
from project_paths import experiment_output, legacy_artifact, require_run_context
from allocation_core import predict_end
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, qc_augment, recover_generator, reference_gain
from code9_core import edge_slope_lifts, parity_disagreements, split_frames, systematic_checks
from educational_fec_demo import encode_block, word_bits

K, N, LIFT = 1280, 2048, 128
SEED = 2026112015
SOURCE = ROOT / 'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
HEADERS = ROOT / 'runs/v008-complete-header-layout-02/artifacts/experiments/E010/results/full_header_layout.json'
CODE5 = ROOT / 'runs/v010-code5-parity-02/artifacts/experiments/E012/results/code5_observations.npz'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def extract(iq, headers, sss, unwrap_slope):
    prefix = word_bits(0x3048)[16:]
    signs = 1 - 2 * encode_block(np.r_[prefix, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    slots = np.arange(2, 302)
    observations, offsets, rows, receiver = [], {-1: [], 1: []}, [], []
    for frame in headers['frames']:
        index = frame['legacy_frame_index']
        h2 = frame['second_header']
        allocations = predict_end(h2['start_resource'], h2['blocks'], h2['layout20']['entries'])['allocations']
        targets = [a for a in allocations if a['code'] == 9]
        if not targets:
            continue
        grid = fft_grid(iq, index)
        equalized = equalize_grid(grid[1:], grid[0], sss, slots)
        data = equalized['nonpilot']
        reserved = equalized['reserved_pilot_ratio']
        pilot_carriers = PILOTS[TRAIN_PILOTS]
        lower_center = pilot_carriers[pilot_carriers < 0].mean()
        upper_center = pilot_carriers[pilot_carriers > 0].mean()
        carrier_mask = ~np.isin(LOADED, PILOTS)
        phase, lifts = edge_slope_lifts(equalized['calibration'][:, 1], LOADED, lower_center, upper_center)
        if unwrap_slope:
            data = data * np.exp(-1j * phase[:, carrier_mask])
            held_carriers = np.setdiff1d(PILOTS, pilot_carriers)
            reserved = reserved * np.exp(-1j * phase[:, np.isin(LOADED, held_carriers)])
        flat = data.ravel()
        gain = reference_gain(flat[:48], signs)
        flat = flat / gain
        receiver.append({'frame': index, 'reserved_pilot_evm': float(np.sqrt(np.mean(abs(reserved - 1) ** 2))),
                         'reserved_pilot_count': int(reserved.size),
                         'reserved_pilot_label_errors': int(np.count_nonzero(abs(np.angle(reserved)) > np.pi / 4)),
                         'baseline_pilot_evm': equalized['reserved_pilot_evm'],
                         'candidate_edge_cycles_by_slot': lifts.tolist(),
                         'wrapped_slope_rad_per_carrier': equalized['calibration'][:, 1].tolist(),
                         'known_prefix_gain': [float(gain.real), float(gain.imag)]})
        for region in targets:
            for index_in_region in range(region['quantity']):
                start = region['start_resource'] + index_in_region * N
                stop = start + N
                if start < 1 or stop + 1 > len(flat):
                    raise ValueError(f'No complete control window for frame {index} resource {start}')
                observations.append(flat[start:stop])
                for shift in offsets:
                    offsets[shift].append(flat[start + shift:stop + shift].real < 0)
                rows.append({'frame': index, 'word': index_in_region, 'start_resource': start})
    return np.asarray(observations), {k: np.asarray(v, dtype=np.uint8) for k, v in offsets.items()}, rows, receiver


def fit_and_check(words, symbols, frames, train):
    u = words[train, :K]
    v = words[train, K:]
    raw_rank = recover_generator(u, np.zeros((len(u), 1), dtype=np.uint8))['rank']
    u_qc, v_qc = qc_augment(u, v, LIFT)
    fit = recover_generator(u_qc, v_qc)
    p = fit.pop('generator')
    fit.update(training_frames=np.unique(frames[train]).tolist(),
               reserved_frames=np.unique(frames[~train]).tolist(),
               training_words=int(train.sum()), reserved_words=int((~train).sum()),
               raw_training_rank=raw_rank, qc_equations=len(u_qc))
    if p is None:
        return fit, None
    predicted = (words[:, :K] @ p) % 2
    err = np.count_nonzero(predicted != words[:, K:], axis=1)
    h = systematic_checks(p)
    syndromes = (words @ h.T) % 2
    if not np.array_equal(syndromes, predicted ^ words[:, K:]):
        raise RuntimeError('Systematic check and parity prediction disagree')
    subgroup = symbols[~train, K:]
    remodulated = 1 - 2 * predicted[~train].astype(float)
    flipped = remodulated.copy()
    flipped[:, ::16] *= -1
    power = np.sum(abs(subgroup) ** 2)
    rows = np.arange(K)
    rotated_rows = (rows // LIFT) * LIFT + (rows + 1) % LIFT
    fit.update(generator_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
               parity_bits_reserved=int((~train).sum()) * (N - K),
               parity_disagreements_reserved=int(err[~train].sum()),
               parity_disagreements_each_word=err.tolist(),
               commutes_with_qc128=bool(np.array_equal(p[rotated_rows], np.roll(p.reshape(K, -1, LIFT), 1, axis=2).reshape(p.shape))),
               reserved_parity_nmse=float(np.sum(abs(subgroup - remodulated) ** 2) / power),
               flipped_reserved_parity_nmse=float(np.sum(abs(subgroup - flipped) ** 2) / power))
    return fit, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unwrap-slope', action='store_true', help='Choose the pilot slope branch along time')
    args = parser.parse_args()
    require_run_context()
    out = experiment_output('E015')
    out.mkdir(parents=True, exist_ok=True)
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    inputs = {path.relative_to(ROOT).as_posix(): digest(path) for path in [SOURCE, HEADERS, CODE5]}
    expected = ['a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
                'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838',
                '854dd5924f3350177f09b1dce97bb78f099175702e204fea3096e348e68dd490']
    if list(inputs.values()) != expected:
        raise RuntimeError('Frozen capture, headers or Code 5 evidence changed')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    iq = np.load(SOURCE, mmap_mode='r', allow_pickle=False)
    symbols, offsets, rows, receiver = extract(iq, json.loads(HEADERS.read_text(encoding='utf-8')), sss, args.unwrap_slope)
    words = (symbols.real < 0).astype(np.uint8)
    frames = np.array([r['frame'] for r in rows])
    published = np.isin(frames, [0, 4, 8])
    randomized, order = split_frames(frames, SEED, train_groups=3)
    published_fit, p = fit_and_check(words, symbols, frames, published)
    randomized_fit, p_random = fit_and_check(words, symbols, frames, randomized)
    result = {'schema': 'starlink-code9-parity/v1', 'input_sha256': inputs, 'sss_sha256': digest(sss_path),
              'code_parameters': {'length': N, 'systematic': K, 'lift': LIFT},
              'receiver': receiver, 'words': rows, 'group_seed': SEED, 'group_order': order,
              'unwrap_slope': args.unwrap_slope,
              'published_groups': published_fit, 'randomized_groups': randomized_fit,
              'same_frozen_generator': p is not None and p_random is not None and np.array_equal(p, p_random),
              'scope': 'Both splits reuse one capture and share frames. This is a published-model replication, with no new RF or true BER.'}
    arrays = {'observed_symbols': symbols, 'hard_bits': words,
              'offset_minus_one_bits': offsets[-1], 'offset_plus_one_bits': offsets[1],
              'published_training': published, 'randomized_training': randomized}
    if p is not None:
        with np.load(CODE5, allow_pickle=False) as prior:
            result['code9_with_code5_errors'] = parity_disagreements(words, prior['parity_generator']).tolist()
            result['code5_with_code9_errors'] = parity_disagreements(prior['hard_bits'], p).tolist()
        result['offset_errors'] = {str(shift): parity_disagreements(bits, p).tolist() for shift, bits in offsets.items()}
        result['whole_word_complement_errors'] = parity_disagreements(1 - words, p).tolist()
        arrays.update(parity_generator=p, systematic_check_matrix=systematic_checks(p))
    if p_random is not None:
        arrays['randomized_parity_generator'] = p_random
    (out / 'code9_parity.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out / 'code9_observations.npz', **arrays)
    for label, fit in [('published', published_fit), ('randomized', randomized_fit)]:
        print(label, fit['training_frames'], 'rank', fit['rank'], 'inconsistencies', fit['inconsistent_equations'],
              'reserved parity disagreement', fit.get('parity_disagreements_reserved'), flush=True)
    if p is None or p_random is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
