"""Timing refinement and crossed geometric channel calibration."""

import numpy as np

from code5_core import LOADED, PERIOD, PILOTS, TRAIN_PILOTS, equalize_grid, sinc_sample
from code9_core import edge_slope_lifts
from code21_ring import evaluate, pack_blocks, recover, reduce_polynomial

SLOTS = np.arange(2, 302)
PILOT_K = PILOTS[TRAIN_PILOTS]
LEFT = float(PILOT_K[PILOT_K < 0].mean())
RIGHT = float(PILOT_K[PILOT_K > 0].mean())
DATA_K = LOADED[~np.isin(LOADED, PILOTS)]
HELD_K = np.setdiff1d(PILOTS, PILOT_K)


def residual_sample_scale(slope_per_slot):
    return -slope_per_slot * 2048 / (2*np.pi*2112)


def corrected_grid(iq, frame, scale_delta):
    """Keep the SSS FFT origin fixed and change the body sample scale."""
    nominal = PERIOD / 640000
    sss_start = 639992 + frame*PERIOD + (2112+64)*nominal
    center = (len(iq)-1)/2
    offsets = np.arange(2048)
    result = []
    for slot in range(1, 302):
        coordinates = sss_start + ((slot-1)*2112+offsets)*(nominal+scale_delta)
        received = sinc_sample(iq, coordinates)
        tau = (coordinates-center)/480e6
        received *= np.exp(-1j*(2*np.pi*48257.986*tau+np.pi*(-1107.157)*tau*tau))
        result.append(np.fft.fft(received)[LOADED % 2048])
    return np.asarray(result)


def evaluate_grid(grid, sss):
    eq = equalize_grid(grid[1:], grid[0], sss, SLOTS)
    slope = eq['calibration'][:, 1]
    phase, cycles = edge_slope_lifts(slope, DATA_K, LEFT, RIGHT)
    unwrapped = slope + cycles*(2*np.pi/(RIGHT-LEFT))
    train = (SLOTS >= 50) & (SLOTS <= 289)
    line = np.polyfit(SLOTS[train], unwrapped[train], 1)
    normalized = eq['nonpilot'] * np.exp(-1j*phase)
    reserved_phase, _ = edge_slope_lifts(slope, HELD_K, LEFT, RIGHT)
    reserved = eq['reserved_pilot_ratio'] * np.exp(-1j*reserved_phase)
    return normalized, {'scale_delta': float(residual_sample_scale(line[0])),
                        'held_pilot_evm': float(np.sqrt(np.mean(abs(reserved-1)**2))),
                        'held_pilot_errors': int(np.count_nonzero(abs(np.angle(reserved)) > np.pi/4)),
                        'held_pilot_count': reserved.size}


def crossfit_gain(observed, labels, split_index, points):
    """Use opposite selector parity for each group and keep its observations disjoint."""
    corrected = observed.copy()
    width = int(labels.max())+1
    for parity in (0, 1):
        use = split_index % 2 == parity
        g = labels[use]
        y = observed[use]
        decisions = points[(abs(y[:, None]-points)**2).argmin(axis=1)]
        numerator = y*decisions.conj()
        real = np.bincount(g, weights=numerator.real, minlength=width)
        imag = np.bincount(g, weights=numerator.imag, minlength=width)
        power = np.bincount(g, weights=abs(decisions)**2, minlength=width)
        gain = np.ones(width, dtype=np.complex128)
        valid = power > 0
        gain[valid] = (real[valid]+1j*imag[valid])/power[valid]
        if np.any(abs(gain[labels[~use]]) < .25):
            raise ValueError('Crossfit gain is too small')
        corrected[~use] /= gain[labels[~use]]
    return corrected


def calibration_consensus(u, v, lift, seed, trials=300):
    """Lift only a unique projected calibration majority, using no reserved words."""
    short = 16
    full_u, full_v = pack_blocks(u, lift), pack_blocks(v, lift)
    us, vs = ([[reduce_polynomial(x, short) for x in row] for row in a] for a in (full_u, full_v))
    rng = np.random.default_rng(seed)
    seen = set()
    eligible = {}
    attempts = 0
    for _ in range(trials):
        chosen = tuple(sorted(rng.choice(len(us), size=len(us[0]), replace=False).tolist()))
        if chosen in seen:
            continue
        seen.add(chosen)
        p, _, _ = recover([us[i] for i in chosen], [vs[i] for i in chosen], short)
        if p is None:
            continue
        attempts += 1
        key = tuple(map(tuple, p))
        if key in eligible:
            eligible[key][1].append(chosen)
        else:
            score = sum(a == b for a, b in zip(evaluate(us, p, short), vs, strict=True))
            if score > len(us)//2:
                eligible[key] = (score, [chosen])
    result = {'trials': trials, 'distinct_trials': len(seen), 'invertible_trials': attempts,
              'projected_majorities': len(eligible), 'full_candidates': []}
    if len(eligible) != 1:
        return None, result
    score, sets = next(iter(eligible.values()))
    result['projected_consensus'] = score
    passed = {}
    for chosen in sets[:16]:
        p, _, _ = recover([full_u[i] for i in chosen], [full_v[i] for i in chosen], lift)
        if p is None:
            continue
        key = tuple(map(tuple, p))
        if key in passed:
            continue
        exact = sum(a == b for a, b in zip(evaluate(full_u, p, lift), full_v, strict=True))
        passed[key] = exact
        result['full_candidates'].append({'fit_indices_within_training': list(chosen), 'training_exact_words': exact})
    majority = [p for p, count in passed.items() if count > len(us)//2]
    result['distinct_complete_majorities'] = len(majority)
    return ([list(r) for r in majority[0]] if len(majority) == 1 else None), result
