"""Measure weak-grid terminal word phase without consulting H2 observations."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from layout_core import NONPILOT
from receiver_models import pn_reference
from ut_receiver_core import capture_grid


SEED = 20261121122
PERIOD = 60
HORIZONS = (1, 2, 4, 8, 16)
CONTROL_COUNT = 31
FIRST_ONSET_SLOT = 11


def deinterleave_slots(data, slots):
    """Use 16 complete code periods and convert to absolute resource phase."""
    if data.shape != (len(slots), len(NONPILOT)):
        raise ValueError('Expected the complete per-slot nonpilot support')
    values = data[:, :16*PERIOD].reshape(len(data), 4, 4, PERIOD)
    offsets = (np.asarray(slots) - 2)*len(NONPILOT)
    indices = (np.arange(PERIOD)[None, :] - offsets[:, None]) % PERIOD
    values = np.take_along_axis(values, indices[:, None, None], axis=-1)
    return values - values.mean(axis=-1, keepdims=True)


def projection_bank(words):
    words = np.asarray(words, dtype=float)
    indices = (np.arange(PERIOD)[None, :] - np.arange(PERIOD)[:, None]) % PERIOD
    centered = words[:, indices] - words.mean(axis=-1)[:, None, None]
    return centered / np.linalg.norm(centered, axis=-1, keepdims=True)


def block_projections(blocks, bank):
    shape = bank.shape[:2]
    fitted = blocks @ bank.reshape(-1, PERIOD).T
    return fitted.reshape(*blocks.shape[:3], *shape)


def select_tail(blocks, bank, fold):
    keep = min(len(blocks), max(HORIZONS))
    blocks = blocks[-keep:]
    projections = block_projections(blocks, bank)
    trial = projections[:, :, fold::2].mean(axis=2)
    received = projections[:, :, 1-fold::2].mean(axis=2)
    train_power = abs(blocks[:, :, fold::2])**2
    test_power = abs(blocks[:, :, 1-fold::2])**2
    lengths = [n for n in HORIZONS if n <= keep]
    fits = np.stack([2*np.sum(abs(trial[-n:])**2, axis=(0, 1))/max(train_power[-n:].sum(), 1e-30)
                     for n in lengths])
    result = []
    for k in range(len(bank)):
        j, phase = np.unravel_index(fits[:, k].argmax(), fits[:, k].shape)
        n = lengths[j]
        gains = trial[-n:, :, k, phase]
        observed = received[-n:, :, k, phase]
        improvement = 2*np.sum(2*(gains.conj()*observed).real-abs(gains)**2)
        best = fits[j, k, phase]
        result.append({'phase': int(phase), 'terminal_slots': n, 'calibration_improvement': float(best),
                       'calibration_ties': int(np.count_nonzero(np.isclose(
                           fits[j, k], best, rtol=1e-10, atol=1e-12))),
                       'held_improvement': float(improvement/max(test_power[-n:].sum(), 1e-30))})
    return result


def slot_metrics(blocks, word, fold):
    projections = (blocks @ word[:, None])[..., 0]
    train = projections[:, :, fold::2].mean(axis=2)
    held = projections[:, :, 1-fold::2].mean(axis=2)
    fit_power = np.sum(abs(blocks[:, :, fold::2])**2, axis=(1, 2, 3))
    held_power = np.sum(abs(blocks[:, :, 1-fold::2])**2, axis=(1, 2, 3))
    fit = 2*np.sum(abs(train)**2, axis=1)/np.maximum(fit_power, 1e-30)
    score = 2*np.sum(2*(train.conj()*held).real-abs(train)**2, axis=1)/np.maximum(held_power, 1e-30)
    return fit, score


def onset_scan(blocks, phase_words, slots, fold):
    fit, held = slot_metrics(blocks, phase_words, fold)
    terminal = max(HORIZONS)
    n = len(slots) - terminal
    # Under uncorrelated noise the gain of one centered code vector per two
    # independent 60-resource blocks explains about 1/118 of train power.
    threshold = .5*(np.median(fit[-terminal:])+1/(2*(PERIOD-1)))
    objective = np.cumsum((fit[:n]-threshold)[::-1])[::-1]
    candidates = np.flatnonzero(np.asarray(slots[:n]) >= FIRST_ONSET_SLOT)
    idx = int(candidates[objective[candidates].argmax()])
    return {'first_periodic_slot': int(slots[idx]), 'threshold': float(threshold),
            'calibration_by_slot': fit.tolist(), 'held_by_slot': held.tolist(),
            'pre_onset_held_median': float(np.median(held[:idx])),
            'post_onset_held_median_before_phase_search': float(np.median(held[idx:n]))}


def check_grid(values, slots, pn, shifted_pn, bank):
    blocks = deinterleave_slots(values*pn[:len(slots)].conj(), slots)
    wrong = deinterleave_slots(values*shifted_pn[:len(slots)].conj(), slots)
    output = []
    for fold in (0, 1):
        chosen, *controls = select_tail(blocks, bank, fold)
        incorrect, = select_tail(wrong, bank[:1], fold)
        upper = max(0., incorrect['held_improvement'], *(c['held_improvement'] for c in controls))
        output.append({'best': chosen, 'wrong_pn': incorrect, 'shuffled': controls,
                       'max_control_held_improvement': upper,
                       'passed': chosen['calibration_ties'] == 1 and chosen['held_improvement'] > upper,
                       'onset': onset_scan(blocks, bank[0, chosen['phase']], slots, fold)})
    return output


def receive(samples, epoch, ratio, frequency, sss):
    grid, slots = capture_grid(samples, epoch, ratio, frequency)
    channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(grid[0]/sss, 2)])
    phase = channel.conj()/np.maximum(abs(channel), 1e-30)
    return grid[1:, ~PILOT]*phase[~PILOT], slots[1:]


def consensus(fits):
    branches = [a for fit in fits for a in fit]
    phases = {a['best']['phase'] for a in branches}
    if len(phases) == 1 and all(a['passed'] for a in branches):
        return phases.pop()
    return None


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    wave = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    source = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    ref = doc('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    word = np.array(ref['cycle_folds'][0]['signed_template'])
    rng = np.random.default_rng(SEED)
    words = np.stack([word, *[rng.permutation(word) for _ in range(CONTROL_COUNT)]])
    bank = projection_bank(words)
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    digest = sha256(sss_path)
    if digest != source['sss_reference_sha256']:
        raise ValueError('Unexpected SSS hash')
    hashes[PROJECT_ROOT / 'references/qin/sssVec.mat'] = digest
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    iq_path = PROJECT_ROOT / SOURCE
    iq = load_samples(iq_path, source)
    hashes[iq_path] = source['source_sha256']
    if not np.array_equal(LOADED[~PILOT], NONPILOT):
        raise ValueError('Unexpected nonpilot ordering')
    slots = np.arange(2, 302)
    pn = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT])
    wrong = np.column_stack([pn_reference(int(k), slots+1) for k in NONPILOT])
    weak = []
    groups = {tick: name for name, values in (
        ('eight', config['folds'][0]['held_ticks']), ('seven', config['folds'][1]['held_ticks']),
        ('overlap', config['overlap_ticks'])) for tick in values}
    for row in wave['offset_grid']['records']:
        tick = row['tick']
        if tick not in groups:
            continue
        sync = row['sync']
        fits = []
        for fold in config['folds']:
            fit = fold['timing']
            values, slots_present = receive(iq, sync['epoch_480_samples'], fit['sampling_scale'],
                                            sync['cfo_hz']+fit['cfo_residual_hz'], sss)
            if slots_present[-1] != 301:
                break
            fits.append(check_grid(values, slots_present, pn, wrong, bank))
        phase = consensus(fits) if len(fits) == len(config['folds']) else None
        weak.append({'tick': tick, 'group': groups[tick], 'last_slot': int(slots_present[-1]),
                     'fits': fits, 'accepted_phase': phase})
        print('weak', tick, phase, [[(a['best']['phase'], round(a['best']['held_improvement'], 4),
                                    round(a['max_control_held_improvement'], 4),
                                    a['onset']['first_periodic_slot']) for a in arms] for arms in fits], flush=True)

    strong = []
    end_by_candidate = {w['candidate']: w['scales']['0.06']['first_binary_resource'] for w in ref['signal_results']}
    for row in source['candidates']:
        index = row['candidate_index']
        if index not in end_by_candidate:
            continue
        values, slots_present = receive(iq, row['pss']['epoch'], row['sampling_scale'], row['cfo_hz'], sss)
        arms = check_grid(values, slots_present, pn, wrong, bank)
        phase = consensus([arms])
        end = end_by_candidate[index]
        strong.append({'candidate': index, 'reference_end': end, 'reference_phase': end % PERIOD,
                       'accepted_phase': phase, 'arms': arms})
        print('strong', index, end % PERIOD, phase, flush=True)
    result = {'schema': 'starlink-ut-weak-tail-phase/v1', 'seed': SEED, 'terminal_horizons_slots': list(HORIZONS),
              'input_sha256': {p.as_posix(): d for p, d in hashes.items()},
              'weak': weak, 'strong_same_iq_controls': strong, 'shuffled_words': words[1:].tolist()}
    destination = experiment_output('E118')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'ut_weak_tail_phase.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                                         encoding='utf-8')


if __name__ == '__main__':
    main()
