"""Find weak-word padding onsets and test the frozen MCS 66 parity rows."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS, WINDOW
from audit_internal_allocations import KINDS, WIDTH
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import capture_grid
from verify_ut_offset_h2_body import map_pn


SEED = 20261121112
GROUP = 32
SYSTEM_66 = SYSTEMATIC_ENDS[66]
LIFT = 512
CONTROL_COUNT = 63


def qam_likelihood(data, gain, points, variance):
    return -abs(data[:, None] - gain[:, None] * points)**2 / variance


def onset_profile(likelihood, labels, pn, end, observed):
    """Log-likelihood gain for each bit cut, including partial-symbol hypotheses."""
    lanes = labels.shape[1]
    count = (end + lanes - 1) // lanes
    logp = likelihood[:count]
    positions = np.arange(count * lanes).reshape(count, lanes)
    truth = pn[positions]
    reference = logsumexp(logp, axis=1)
    per_offset = []
    for offset in range(lanes):
        known = (positions < end) & (np.arange(lanes) >= offset)
        allowed = np.all((truth[:, None] == labels[None]) | ~known[:, None], axis=2)
        ratio = logsumexp(np.where(allowed, logp, -np.inf), axis=1) - reference + known.sum(axis=1) * np.log(2.)
        per_offset.append(np.where(observed[:count], ratio, 0.))
    after = np.r_[np.cumsum(per_offset[0][::-1])[::-1][1:], 0.]
    scores = (np.stack(per_offset, axis=1) + after[:, None]).ravel()
    return np.r_[scores[:end], 0.]


def estimate_padding(data, channel, points, labels, pn, end, rng):
    lanes = labels.shape[1]
    first = end // lanes - WINDOW
    train = rng.choice(WINDOW // GROUP, size=2, replace=False)
    ref = map_pn(pn[first * lanes:(first + WINDOW) * lanes].reshape(-1, lanes), points, labels)
    index = first + np.arange(WINDOW)
    use = np.isin(np.arange(WINDOW) // GROUP, train)
    model = channel[index] * ref
    gain = np.vdot(model[use], data[index][use]) / np.vdot(model[use], model[use])
    residual = data[index][use] - gain * model[use]
    variance = float(np.mean(abs(residual)**2))
    support = np.arange(first) // GROUP
    keep = rng.permutation(np.unique(support))
    chosen = keep[:len(keep) // 2]
    observed = np.isin(np.arange(len(data)) // GROUP, chosen) & (np.arange(len(data)) < first)
    held = ~observed & (np.arange(len(data)) < first)
    likelihood = qam_likelihood(data, gain * channel, points, variance)
    fold_rows = []
    for trained, tested in ((observed, held), (held, observed)):
        score = onset_profile(likelihood, labels, pn, end, trained)
        # Whole held groups create flat likelihood intervals. Keep the full tied range.
        near = np.flatnonzero(np.isclose(score, score.max(), rtol=0, atol=1e-9))
        low, high = int(near[0]), int(near[-1])
        retained = onset_profile(likelihood, labels, pn, end, tested)
        shifted = [onset_profile(likelihood, labels, np.roll(pn, step), end, tested) for step in (-1, 1)]
        fold_rows.append({'onset_bit_interval': [low, high], 'fit_log_ratio': float(score[high]),
                          'held_log_ratio': float(retained[high]), 'held_wrong_pn': [float(s[high]) for s in shifted]})
    full = onset_profile(likelihood, labels, pn, end, observed | held)
    safe = max(row['onset_bit_interval'][1] for row in fold_rows) + GROUP * lanes
    if safe >= first * lanes or not all(row['held_log_ratio'] > 0 for row in fold_rows):
        safe = first * lanes
    s = data[index][~use]
    t = (gain * model)[~use]
    anchor = float(np.vdot(t, s).real / (np.linalg.norm(t) * np.linalg.norm(s)))
    return {'gain': [float(gain.real), float(gain.imag)], 'noise_proxy': variance,
            'anchor_held_cosine': anchor, 'calibration_groups': train.tolist(), 'onset_first_fold_groups': chosen.tolist(),
            'joint_best_bit': int(np.argmax(full)), 'padding_start_for_conditioning': safe, 'folds': fold_rows}, likelihood


def xor_moments(likelihood, labels, pn=None, known=None):
    lanes = labels.shape[1]
    supported = likelihood
    if known is not None:
        mask = np.all((pn[:, None] == labels[None]) | ~known[:, None], axis=2)
        supported = np.where(mask, supported, -np.inf)
    states = ((np.arange(1 << lanes)[:, None] >> np.arange(lanes)) & 1)
    signs = 1 - 2 * ((labels @ states.T) % 2).astype(float)
    posterior = np.exp(supported - logsumexp(supported, axis=1, keepdims=True))
    return posterior @ signs


def parity_prediction(moments, rows, first_parity):
    predicted, observed = [], []
    for row in rows:
        groups, masks = np.unique(row // 4, return_inverse=True)
        combined = np.zeros(len(groups), dtype=int)
        np.bitwise_xor.at(combined, masks, 1 << (row % 4))
        if groups[-1] * 4 < first_parity:
            raise ValueError('Expected a parity symbol at the end of each check row')
        atoms = moments[groups, combined]
        predicted.append(np.prod(atoms[:-1]))
        observed.append(atoms[-1])
    return np.asarray(predicted), np.asarray(observed)


def signed_cosine(x, y):
    return float((x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def parity_audit(likelihood, labels, pn, support, onset, rotations):
    raw = xor_moments(likelihood, labels)
    coords = np.arange(4 * len(likelihood)).reshape(-1, 4)
    known = (coords >= onset) & (coords < SYSTEM_66)
    padded = xor_moments(likelihood, labels, pn[coords], known)
    shifted = np.where(support < SYSTEM_66, (support // LIFT) * LIFT + (support + 1) % LIFT, support)
    outcome = {}
    for name, moments in (('raw', raw), ('pn_conditioned', padded)):
        predict, held = parity_prediction(moments, support, SYSTEM_66)
        wrong, _ = parity_prediction(moments, shifted, SYSTEM_66)
        observed = held.reshape(-1, LIFT)
        outcome[name] = {'cosine': signed_cosine(predict, held), 'shifted_systematic_cosine': signed_cosine(wrong, held),
                         'predicted_norm': float(np.linalg.norm(predict)), 'held_norm': float(np.linalg.norm(held)),
                         'parity_rotation_controls': [signed_cosine(predict, np.roll(observed, int(s), axis=1).ravel()) for s in rotations],
                         'mean_hard_parity': float(np.mean(np.sign(predict) * np.sign(held)))}
    return outcome


def main():
    require_run_context()
    hashes = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(art(run, exp, name))

    headers, confirmation = [doc('v108-ut-offset-body-first' + suffix, 'E110', 'ut_offset_body_first.json') for suffix in ('', '-confirm')]
    if headers != confirmation:
        raise ValueError('H2 runs differ')
    clocks = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')['folds']
    geometry = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')['offset_grid']['records']
    sync = {row['tick']: row['sync'] for row in geometry if 'sync' in row}
    capture = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    path = PROJECT_ROOT / SOURCE
    samples = load_samples(path, capture)
    hashes[path] = capture['source_sha256']
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    digest = sha256(sss_path)
    if digest != capture['sss_reference_sha256']:
        raise ValueError('Unexpected SSS reference')
    hashes[PROJECT_ROOT / 'references/qin/sssVec.mat'] = digest
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    model = doc('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json')['fit']['winner']
    pn = equality_predictions(np.arange(16385), pn_bits(), model)
    with np.load(art('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as source:
        modes = {'16QAM': (source['constellation_points'], source['frozen_gray_mapping'])}
        strong = source['mcs66_refined'][:16, 1:-1]
    mapped = doc('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')['cross32_mapping']
    modes['cross32'] = (np.array([complex(item['i'], item['q']) for item in mapped]) / np.sqrt(20),
                        labels_as_bits(np.array([item['label'] for item in mapped]), 5))
    with np.load(art('v085-mcs66-window-check', 'E085', 'mcs66_phase_window_check.npz'), allow_pickle=False) as source:
        matrix = source['row_support']
    rotors = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in LOADED[~PILOT]])
    rng = np.random.default_rng(SEED)
    rotations = rng.choice(np.arange(1, LIFT), size=CONTROL_COUNT, replace=False)
    rows, parity = [], []
    for frame in headers['rows']:
        head = frame['primary']['selected']
        if not head or not head['crc_ok']:
            continue
        tick = frame['tick']
        for fold, clock in enumerate(clocks):
            fit = clock['timing']
            anchor = sync[tick]
            grid, slots = capture_grid(samples, anchor['epoch_480_samples'], fit['sampling_scale'], anchor['cfo_hz'] + fit['cfo_residual_hz'])
            channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(grid[0] / sss, 2)])[~PILOT]
            data = (grid[1:, ~PILOT] * rotors[:len(slots) - 1].conj()).ravel()
            broad = np.tile(channel, len(slots) - 1)
            start = 8420 + head['coding_length']
            for entry in head['layout']['entries']:
                code, count = entry['mcs_candidate'], entry['quantity']
                width, end = WIDTH[code], SYSTEMATIC_ENDS[code]
                stop = start + width * count
                spec = modes[KINDS[code]]
                part = slice(stop - width, stop)
                # Use identical OFDM resource partitions in the two timing replays.
                word_rng = np.random.default_rng(SEED + 100 * tick + code)
                analysis, likelihood = estimate_padding(data[part], broad[part], *spec, pn, end, word_rng)
                rows.append({'tick': tick, 'fold': fold, 'code': code, 'quantity': count, 'start': start,
                             'stop': stop, 'systematic_bits': end, **analysis})
                if code in (66, 70):
                    cutoff = analysis['padding_start_for_conditioning']
                    gain, noise = complex(*analysis['gain']), analysis['noise_proxy']
                    indices = range(count) if code == 66 else (count - 1,)
                    for index in indices:
                        local = slice(start + index * width, start + (index + 1) * width)
                        received = likelihood if index == count - 1 else qam_likelihood(data[local], gain * broad[local], spec[0], noise)
                        onset = cutoff if index == count - 1 else SYSTEM_66
                        parity.append({'tick': tick, 'fold': fold, 'code': code, 'word': index, 'known_cutoff': onset,
                                       **parity_audit(received, spec[1], pn, matrix, onset, rotations)})
                start = stop
    points, labels = modes['16QAM']
    positive = [parity_audit(qam_likelihood(word, np.ones(len(word)), points, .03), labels, pn, matrix, SYSTEM_66, rotations)['raw']
                for word in strong]
    result = {'schema': 'starlink-weak-padding-and-parity/v1', 'seed': SEED, 'rotation_offsets': rotations.tolist(),
              'inputs_sha256': {p.as_posix(): digest for p, digest in hashes.items()}, 'padding': rows,
              'weak_check_prediction': parity, 'strong_positive_replay': positive}
    out = experiment_output('E111')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ut_weak_padding_and_parity.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('padding estimates', [(r['tick'], r['code'], r['fold'], r['joint_best_bit']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
