"""Find conditional weak-grid tail edges with held frequency blocks."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_offset_grid_pilots import PILOT
from audit_ut_repeated_data import checked_artifact, read
from audit_weak_tail_paths import onset_range
from audit_weak_tail_phase import HORIZONS, receive
from code5_core import LOADED
from layout_core import NONPILOT
from receiver_models import pn_reference


SEED = 20261121123
PERIOD = 60
BLOCK = 32
MIN_ANCHOR = 6
MAX_ANCHOR = 16
CHANNEL_DEGREES = (0, 1)
MAX_BRANCH_SPREAD = 4*PERIOD


def fit_anchor(data, signs, slots, query_slots, phase_degree):
    """Estimate per-carrier gain on tail-only slots outside the edge scan."""
    reference = data*signs
    constant = reference.mean(axis=0)
    if phase_degree == 0:
        return np.broadcast_to(constant, (len(query_slots), len(constant)))
    center = np.mean(slots)
    slot_gain = reference @ constant.conj()
    phase_slope = np.polyfit(slots-center, np.unwrap(np.angle(slot_gain)), 1)[0]
    detrended = (np.exp(-1j*phase_slope*(slots-center))[:, None] * reference).mean(axis=0)
    return np.exp(1j*phase_slope*(query_slots-center))[:, None] * detrended


def remove_block_mean(values):
    bins = np.arange(values.shape[-1])//BLOCK
    starts, counts = np.unique(bins, return_index=True, return_counts=True)[1:]
    sums = np.add.reduceat(values, starts, axis=-1)
    return values - (sums/counts)[..., bins]


def scan_endpoints(data, waveform, coordinates, candidates, fit):
    centered = remove_block_mean(data)
    prediction = remove_block_mean(waveform[None]*(coordinates[None] >= candidates[:, None, None]))
    reduction = 2*(prediction.conj()*centered[None]).real - abs(prediction)**2
    return np.stack([reduction[..., part].sum(axis=(1, 2))/max(np.sum(abs(centered[:, part])**2), 1e-30)
                     for part in (fit, ~fit)])


def cross_score(scores, candidates):
    arms = []
    for k in (0, 1):
        values = scores[k]
        i = int(values.argmax())
        ties = np.count_nonzero(np.isclose(values, values[i], rtol=1e-10, atol=1e-12))
        other = scores[1-k]
        advantage = other[i]-max(other[0], other[-1])
        arms.append({'selected': int(candidates[i]), 'ties': int(ties), 'calibration': float(values[i]),
                     'complement_score': float(other[i]), 'complement_margin_vs_outer_edges': float(advantage),
                     'complement_max': float(other.max())})
    return arms


def retained_bounds(arms, candidates):
    records = [a for receiver in arms for degree in receiver for a in degree['arms']]
    choices = np.array([a['selected'] for a in records])
    safe = all(a['ties'] == 1 and a['complement_margin_vs_outer_edges'] > 0 for a in records)
    keep = (safe and choices.min() > candidates[0] and choices.max() < candidates[-1]
            and np.ptp(choices) <= MAX_BRANCH_SPREAD)
    bound = [int(choices.min()-PERIOD), int(choices.max()+PERIOD)] if keep else None
    return {'retained_interval': bound, 'spread': int(np.ptp(choices)),
            'min_selected': int(choices.min()), 'max_selected': int(choices.max()),
            'all_arms_same': bool(np.ptp(choices) == 0), 'cross_holdout_positive': safe}


def mask_bins(rng, width):
    group = np.arange(width)//BLOCK
    count = int(group[-1]+1)
    if count % 2:
        raise ValueError('Frequency allocation expects paired groups')
    orientation = rng.integers(0, 2, size=count//2)
    return group % 2 == orientation[group//2]


def edge_geometry(row):
    interval = onset_range(row)
    if interval is None:
        return None
    arms = [a['onset'] for fit in row['fits'] for a in fit]
    if not all(a['post_onset_held_median_before_phase_search'] > a['pre_onset_held_median'] for a in arms):
        return None
    width = len(NONPILOT)
    lo, hi = interval
    phase = row['accepted_phase']
    candidates = np.arange(lo+(phase-lo) % PERIOD, hi+1, PERIOD)
    scan = np.arange(lo//width-1, hi//width+1)
    upper = 300-max(HORIZONS)
    anchor = np.arange(scan[-1]+2, min(upper, scan[-1]+2+MAX_ANCHOR))
    background = scan-scan[-1]+scan[0]-3
    if len(anchor) < MIN_ANCHOR or background[0] < 0:
        return None
    return candidates, scan, anchor, background


def analyze_grid(y, word, phase, geometry, fit, injected):
    candidates, scan, anchor, background = geometry
    width = len(NONPILOT)
    coords = scan[:, None]*width+np.arange(width)
    signs = word[(coords-phase) % PERIOD]
    anchor_signs = word[(anchor[:, None]*width+np.arange(width)-phase) % PERIOD]
    gains = [fit_anchor(y[anchor], anchor_signs, anchor, scan, degree)
             for degree in CHANNEL_DEGREES]
    model = gains[0]*signs
    actual = []
    test = {float(a): [] for a in (.5, 1.)}
    for degree, gain in zip(CHANNEL_DEGREES, gains, strict=True):
        template = gain*signs
        scores = scan_endpoints(y[scan], template, coords, candidates, fit)
        actual.append({'phase_degree': degree, 'scores': scores.tolist(), 'arms': cross_score(scores, candidates)})
        for scale in test:
            mix = y[background] + scale*model*(coords >= injected)
            scores = scan_endpoints(mix, scale*template, coords, candidates, fit)
            test[scale].append({'phase_degree': degree, 'arms': cross_score(scores, candidates)})
    return actual, test


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    a, b = [doc('v116-ut-weak-tail-phase-final'+suffix, 'E118', 'ut_weak_tail_phase.json')
            for suffix in ('', '-confirm')]
    if a != b:
        raise ValueError('Weak tail phase runs disagree')
    grid = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')['offset_grid']['records']
    sync = {row['tick']: row['sync'] for row in grid if 'sync' in row}
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    reference = doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    source = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    word = np.array(reference['cycle_folds'][0]['signed_template'])
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] = sha256(sss_path)
    if hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] != source['sss_reference_sha256']:
        raise ValueError('Unexpected SSS hash')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    iq_path = PROJECT_ROOT/SOURCE
    iq = load_samples(iq_path, source)
    hashes[iq_path] = source['source_sha256']
    if not np.array_equal(LOADED[~PILOT], NONPILOT):
        raise ValueError('Unexpected nonpilot order')
    rx_slots = np.arange(2, 302)
    pn = np.column_stack([pn_reference(int(k), rx_slots) for k in NONPILOT])
    rng = np.random.default_rng(SEED)
    results = []
    for row in a['weak']:
        tick = row['tick']
        base = {'tick': tick, 'phase': row['accepted_phase'], 'coarse_interval': onset_range(row)}
        geometry = edge_geometry(row)
        if geometry is None:
            results.append({**base, 'retained_interval': None, 'fits': []})
            continue
        candidates, scan, anchor, background = geometry
        fit = mask_bins(rng, len(NONPILOT))
        n = len(candidates)
        injected = int(rng.choice(candidates[n//4:3*n//4]))
        evidence = []
        controls = {float(v): [] for v in (.5, 1.)}
        for fold in config['folds']:
            timing = fold['timing']
            target = sync[tick]
            values, slots = receive(iq, target['epoch_480_samples'], timing['sampling_scale'],
                                    target['cfo_hz']+timing['cfo_residual_hz'], sss)
            if not np.array_equal(slots, rx_slots):
                raise ValueError('Tail-qualified tick is incomplete')
            results_raw, synthetic = analyze_grid(values*pn.conj(), word, row['accepted_phase'], geometry, fit, injected)
            evidence.append(results_raw)
            for scale in controls:
                controls[scale].append(synthetic[scale])
        summary = retained_bounds(evidence, candidates)
        injections = [{'amplitude': scale, 'injected_endpoint': injected,
                       **retained_bounds(measures, candidates), 'fits': measures}
                      for scale, measures in controls.items()]
        results.append({**base, **summary, 'scan_slots': (scan+2).tolist(), 'anchor_slots': (anchor+2).tolist(),
                        'background_slots': (background+2).tolist(), 'endpoint_candidates': candidates.tolist(),
                        'first_arm_blocks': np.unique(np.flatnonzero(fit)//BLOCK).tolist(),
                        'fits': evidence, 'injections': injections})
        print(tick, row['accepted_phase'], summary, [i['retained_interval'] for i in injections], flush=True)
    output = {'schema': 'starlink-weak-tail-endpoint/v1', 'seed': SEED, 'block_width': BLOCK,
              'max_branch_spread': MAX_BRANCH_SPREAD, 'input_sha256': {p.as_posix(): h for p, h in hashes.items()},
              'rows': results}
    destination = experiment_output('E119')
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'ut_weak_tail_endpoint.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n',
                                                         encoding='utf-8')


if __name__ == '__main__':
    main()
