"""Score frozen 66/70 parity families on complementary raw weak observations."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, PILOTS, TRAIN_PILOTS
from receiver_models import pn_reference
from ut_receiver_core import capture_grid
from verify_ut_offset_h2_body import predicted_markers


SEED = 20261121127
LIFT = 512
WORD_RESOURCES = 4096
GROUP = 5
TIME_SMOOTHING_SLOTS = 2.5


def extrinsic_bits(moments, support):
    evidence = moments.ravel()[support]
    return np.prod(evidence[:, :-1], axis=1), evidence[:, -1]


def bit_moments(y, channel, noise, points, labels):
    logp = -abs(y[:, None]-channel[:, None]*points[None])**2/noise[:, None]
    weights = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
    return weights @ (1-2*labels.astype(float))


def pilot_model(grid, rotor, sss, subset):
    data = grid[1:]*rotor
    base = np.concatenate([gaussian_filter1d(v, SMOOTH_BINS) for v in np.split(grid[0]/sss, 2)])
    predicted = np.empty_like(data)
    variance = np.empty(data.shape, dtype=float)
    measured, forecast = [], []
    for side in (LOADED < 0, LOADED > 0):
        pilots = np.flatnonzero(PILOT & side)
        train, check = pilots[subset[side[PILOT]]], pilots[~subset[side[PILOT]]]
        fit_y = data[:, train]
        h = base[train]
        score = gaussian_filter1d(fit_y @ h.conj(), TIME_SMOOTHING_SLOTS)/max(np.vdot(h,h).real, 1e-30)
        predicted[:, side] = score[:, None]*base[side]
        residual = abs(fit_y-score[:, None]*h)**2
        local = gaussian_filter1d(residual.mean(axis=1), TIME_SMOOTHING_SLOTS)
        variance[:, side] = np.maximum(local, max(1e-30, .05*np.median(local)))[:, None]
        measured.append(data[:, check].ravel())
        forecast.append((score[:, None]*base[check]).ravel())
    observed = np.concatenate(measured)
    expected = np.concatenate(forecast)
    pilot_quality = signed_cosine(observed, expected)
    return [a[:, ~PILOT].ravel() for a in (data, predicted, variance)], pilot_quality


def signed_cosine(x, y):
    return float(np.vdot(x, y).real/max(np.linalg.norm(x)*np.linalg.norm(y), 1e-30))


def weak_scores(moments, support, selected_groups, shifts):
    predictions, held = [np.asarray(r).reshape(len(moments), -1, LIFT) for r in zip(
        *(extrinsic_bits(word, support) for word in moments), strict=True)]
    use = np.isin(np.arange(len(moments))//GROUP, selected_groups)
    arms = []
    for selection in (use, ~use):
        if not selection.any():
            continue
        a, b = predictions[selection], held[selection]
        actual = signed_cosine(a, b)
        null = [signed_cosine(a, np.roll(b, shift, axis=-1)) for shift in shifts]
        arms.append({'words': np.flatnonzero(selection).tolist(), 'held_cosine': actual,
                     'checks': int(a.size), 'prediction_rms': float(np.sqrt(np.mean(a*a))),
                     'null_ranks_ge_main': sum(x>=actual for x in null),
                     'wrong_target_cosines': null, 'pass': actual>max(0., *null)})
    return arms


def allocations(older, new):
    targets = []
    for row in older['rows']:
        a, b = row['primary']['selected'], row['other_pilot_fit']['selected']
        if not a or not b or not (a['crc_ok'] and b['crc_ok']) or a['words_hex'] != b['words_hex']:
            continue
        for record in predicted_markers(8420+a['coding_length'], a['layout']['entries']):
            if record['mcs'] in (66, 70):
                targets.append({'tick': row['tick'], 'source': 'E110_verified', **record})
    row, = [r for r in new['rows'] if r['tick'] == 14]
    joint = row['trials'][0]['methods']['joint']
    if not joint['common_crc_valid']:
        raise ValueError('No frozen E119 candidate')
    a = joint['primary']['selected']
    marker = predicted_markers(8420+a['coding_length'],a['layout']['entries'])[0]
    targets.append({'tick': row['tick'], 'source': 'E119_candidate', **marker})
    return targets


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    p, q = [artifact('v118-mcs70-sparse'+s, 'E120', 'mcs70_sparse_checks.npz') for s in ('','-confirm')]
    if hashes[p] != hashes[q]:
        raise ValueError('Recovered check matrices differ')
    with np.load(p, allow_pickle=False) as z:
        support70 = np.stack([np.flatnonzero(row) for row in np.unpackbits(z['check_matrix_packed'], axis=1)])
    with np.load(artifact('v081-mcs66-systematic-checks', 'E083', 'mcs66_partial_checks.npz'), allow_pickle=False) as z:
        support66 = np.sort(z['row_support'], axis=1)
    by_format = {66: support66, 70: support70}
    for support in by_format.values():
        if not np.all(np.diff(support//4, axis=1)>0) or len(support) != 2*LIFT:
            raise ValueError('Expected two check families with distinct QAM symbols')
    old = doc('v108-ut-offset-body-first', 'E110', 'ut_offset_body_first.json')
    new = doc('v117-ut-weak-two-allocations', 'E119', 'ut_weak_tail_two_allocations.json')
    jobs = allocations(old, new)
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    source = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    capture = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')
    sync = {row['tick']: row['sync'] for row in capture['offset_grid']['records'] if 'sync' in row}
    with np.load(artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as z:
        points, labels = z['constellation_points'], z['frozen_gray_mapping']
    path = legacy_artifact('reference/qin/sssVec.mat')
    hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] = sha256(path)
    if hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] != source['sss_reference_sha256']:
        raise ValueError('Unexpected SSS hash')
    sss = loadmat(path)['sssVecFull'].ravel()[LOADED % 1024]
    raw = PROJECT_ROOT/SOURCE
    iq = load_samples(raw, source)
    hashes[raw] = source['source_sha256']
    slots = np.arange(2, 302)
    pn = np.column_stack([pn_reference(int(k), slots) for k in LOADED]).conj()
    subset = np.isin(np.arange(len(PILOTS)), TRAIN_PILOTS)
    rng = np.random.default_rng(SEED)
    rotations = rng.choice(np.arange(1,LIFT), 63, replace=False)
    for job in jobs:
        tick = job['tick']
        lo, hi = job['allocation_start'], job['word_stop']
        use = slice(lo,hi)
        groups = np.unique(np.arange(job['count'])//GROUP)
        chosen = rng.permutation(groups)[:max(1,len(groups)//2)]
        fits = []
        for offset in config['folds']:
            timing = offset['timing']
            match = sync[tick]
            grid, actual_slots = capture_grid(iq, match['epoch_480_samples'], timing['sampling_scale'],
                                              match['cfo_hz']+timing['cfo_residual_hz'])
            branches = []
            for selected in (subset, ~subset):
                values, quality = pilot_model(grid,pn[:len(actual_slots)-1],sss,selected)
                moments = bit_moments(*(v[use] for v in values),points,labels).reshape(-1,WORD_RESOURCES,4)
                scores = {name: weak_scores(moments, support, chosen, rotations) for name, support in by_format.items()}
                branches.append({'held_pilot_cosine': quality,'format_scores': scores})
            fits.append(branches)
        job.update(train_word_groups=chosen.tolist(),fits=fits)
        print(tick, job['mcs'], job['count'],
              [[(round(v['held_pilot_cosine'],4),
                 {k:[(round(a['held_cosine'],5),a['null_ranks_ge_main']) for a in result]
                  for k,result in v['format_scores'].items()}) for v in sub] for sub in fits], flush=True)
    output = {'schema': 'starlink-weak-code66-code70-transfer/v1', 'seed': SEED,
              'rotation_controls': rotations.tolist(), 'input_sha256': {p.as_posix(): h for p, h in hashes.items()},
              'rows': jobs}
    out = experiment_output('E120')
    out.mkdir(parents=True, exist_ok=True)
    (out/'weak_66_70_check_transfer.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
