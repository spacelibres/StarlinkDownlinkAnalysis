"""Test PN15 continuation unique to the longer 16QAM systematic aperture."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import capture_grid
from verify_ut_offset_h2_body import map_pn, predicted_markers


SEED = 20261121126
SHORT_END = 3072
LONG_END = 3200
APERTURE = 128
CONTROL_COUNT = 63


def pn_likelihood_ratio(data, gain, points, expected, variance):
    known = -abs(data-gain*expected)**2/variance
    unknown = -abs(data[:, None]-gain[:, None]*points[None])**2/variance
    return known - logsumexp(unknown, axis=1) + np.log(len(points))


def probe_word(data, channel, pn, points, labels, orders, train):
    start = SHORT_END-APERTURE
    stop = LONG_END+APERTURE
    expected = map_pn(pn[start*4:stop*4].reshape(-1, 4), points, labels)
    wrong = [map_pn(pn[start*4+shift:stop*4+shift].reshape(-1, 4), points, labels) for shift in (-1, 1)]
    y, h = data[start:stop], channel[start:stop]
    used = np.isin(np.arange(APERTURE)//32, train)
    contested = slice(APERTURE, 2*APERTURE)
    post = slice(2*APERTURE, 3*APERTURE)
    scores = []
    for fit in (used, ~used):
        anchor = h[:APERTURE][fit]*expected[:APERTURE][fit]
        target = y[:APERTURE][fit]
        gain = np.vdot(anchor, target)/max(np.vdot(anchor, anchor).real, 1e-30)
        noise = max(float(np.mean(abs(target-gain*anchor)**2)), 1e-30)
        coefficients = h*gain
        value = pn_likelihood_ratio(y, coefficients, points, expected, noise)
        alternatives = [pn_likelihood_ratio(y, coefficients, points, w, noise) for w in wrong]
        template = expected[contested].reshape(-1, 16)
        shuffled = [float(pn_likelihood_ratio(y[contested], coefficients[contested], points,
                                              template[order].ravel(), noise).sum()) for order in orders]
        main = float(value[contested].sum())
        anchor_held = float(value[:APERTURE][~fit].sum())
        continuation = float(value[post].sum())
        wrong_sums = [float(a[contested].sum()) for a in alternatives]
        scores.append({'known_gain': [float(gain.real), float(gain.imag)], 'noise': noise,
                       'anchor_held_nats': anchor_held, 'contested_nats': main,
                       'after_long_endpoint_nats': continuation, 'wrong_bit_offset_nats': wrong_sums,
                       'shuffled_nats': shuffled,
                       'shuffles_ge_observed': sum(s >= main for s in shuffled),
                       'supports_70_continuation': anchor_held > 0 and main > max(0., *wrong_sums, *shuffled),
                       'supports_stop_at_short_endpoint': anchor_held > 0 and main < 0,
                       'anchor_hold_resources': int((~fit).sum())})
    return scores


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    old = doc('v108-ut-offset-body-first', 'E110', 'ut_offset_body_first.json')
    a, b = [doc('v117-ut-weak-two-allocations'+s, 'E119', 'ut_weak_tail_two_allocations.json') for s in ('','-confirm')]
    if a != b:
        raise ValueError('E119 candidate receipts disagree')
    h2 = doc('v116-ut-weak-tail-paths-markers', 'E118', 'ut_weak_tail_paths.json')['h2_start']
    targets = []
    for row in old['rows']:
        first, other = row['primary']['selected'], row['other_pilot_fit']['selected']
        if not first or not other or not first['crc_ok'] or not other['crc_ok'] or first['words_hex'] != other['words_hex']:
            continue
        markers = predicted_markers(h2+first['coding_length'], first['layout']['entries'])
        targets.extend({'tick': row['tick'], 'candidate_mcs': m['mcs'], 'start': m['word_stop']-4096,
                        'source': 'E110_crc_and_body'} for m in markers if m['mcs'] in (66,70))
    row, = [r for r in a['rows'] if r['tick']==14 and r['trials'][0]['methods']['joint']['common_crc_valid']]
    head = row['trials'][0]['methods']['joint']['primary']['selected']
    first = predicted_markers(h2+head['coding_length'], head['layout']['entries'])[0]
    targets.append({'tick': row['tick'], 'candidate_mcs': first['mcs'], 'start': first['word_stop']-4096,
                    'source': 'E119_crc_candidate'})
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    model = doc('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json')['fit']['winner']
    pn = equality_predictions(np.arange(16384), pn_bits(), model)
    with np.load(artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as data:
        points, labels = data['constellation_points'], data['frozen_gray_mapping']
    source = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    weak = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')['offset_grid']['records']
    anchors = {row['tick']: row['sync'] for row in weak if 'sync' in row}
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    digest = sha256(sss_path)
    if digest != source['sss_reference_sha256']:
        raise ValueError('Unexpected SSS hash')
    hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] = digest
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    path = PROJECT_ROOT/SOURCE
    iq = load_samples(path, source)
    hashes[path] = source['source_sha256']
    slots = np.arange(2, 302)
    rot = np.column_stack([pn_reference(int(k), slots) for k in LOADED[~PILOT]]).conj()
    rng = np.random.default_rng(SEED)
    orders = np.stack([rng.permutation(APERTURE//16) for _ in range(CONTROL_COUNT)])
    for target in targets:
        tick, start = target['tick'], target['start']
        assignment = rng.permutation(4)[:2]
        arms = []
        for fold in config['folds']:
            timing = fold['timing']
            s = anchors[tick]
            grid, rx_slots = capture_grid(iq, s['epoch_480_samples'], timing['sampling_scale'],
                                          s['cfo_hz']+timing['cfo_residual_hz'])
            raw = (grid[1:, ~PILOT]*rot[:len(rx_slots)-1]).ravel()
            occupancy = np.concatenate([gaussian_filter1d(v, SMOOTH_BINS) for v in np.split(grid[0]/sss, 2)])
            channel = np.tile(occupancy[~PILOT], len(rx_slots)-1)
            arms.append(probe_word(raw[start:start+4096], channel[start:start+4096], pn, points, labels, orders, assignment))
        target.update(anchor_first_groups=assignment.tolist(),
                      short_endpoint=start+SHORT_END, long_endpoint=start+LONG_END, fits=arms)
        print(tick, target['candidate_mcs'], [(round(z['anchor_held_nats'], 3), round(z['contested_nats'], 3),
               round(z['after_long_endpoint_nats'], 3), z['shuffles_ge_observed'])
              for pair in arms for z in pair], flush=True)
    output = {'schema': 'starlink-weak-66-70-pn-continuation/v1', 'seed': SEED, 'template_orders': orders.tolist(),
              'input_sha256': {p.as_posix(): h for p, h in hashes.items()}, 'rows': targets}
    out = experiment_output('E120')
    out.mkdir(parents=True, exist_ok=True)
    (out/'weak_mcs66_mcs70.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
