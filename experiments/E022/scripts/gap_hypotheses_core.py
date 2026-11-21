"""Finite sequence and GF(2) prediction tests for unassigned coordinates."""

import numpy as np


def trajectory(sequence, indices, phase, direction, complement):
    return sequence[(phase+direction*indices) % len(sequence)] ^ int(complement)


def phase_fit(sequence, indices, bits):
    """Find the best shared phase, traversal direction, and polarity."""
    ref = np.fft.rfft(1 - 2*sequence.astype(float))
    polarity = 1 - 2*bits.astype(float).ravel()
    winner = None
    for direction in (1,-1):
        bins = np.bincount((direction*indices.ravel())%len(sequence), weights=polarity, minlength=len(sequence))
        correlation = np.fft.irfft(ref*np.fft.rfft(bins).conj(), n=len(sequence))
        score = np.rint(correlation).astype(int)
        phase = int(np.argmax(abs(score)))
        hits = (bits.size+abs(int(score[phase])))//2
        if winner is None or hits > winner['training_hits']:
            winner = {'phase':phase, 'direction':direction, 'complement':bool(score[phase]<0),
                      'training_hits':hits}
    return winner


def histories(bits, order):
    shape = (len(bits), bits.shape[1]-order)
    pattern = np.zeros(shape, dtype=np.uint32)
    for i in range(order):
        pattern |= bits[:, i:i+shape[1]].astype(np.uint32) << i
    return pattern


def walsh_scores(states, targets, order):
    score = np.bincount(states.ravel(), weights=(1-2*targets.astype(float)).ravel(),
                        minlength=1<<order).astype(np.int64)
    step = 1
    while step < len(score):
        block = score.reshape(-1,2*step)
        first, second = block[:,:step].copy(), block[:,step:].copy()
        block[:,:step] = first+second
        block[:,step:] = first-second
        step *= 2
    return score


def recurrence_fit(bits, order=16, target=None):
    target = bits[:,order:] if target is None else target
    score = walsh_scores(histories(bits, order), target, order)
    mask = int(np.argmax(abs(score)))
    return {'mask':mask,'intercept':bool(score[mask]<0),
            'lags':[order-j for j in range(order) if (mask>>j)&1],
            'training_hits':(target.size+abs(int(score[mask])))//2,
            'training_total':target.size,
            'tied_masks':int(np.count_nonzero(abs(score)==abs(score[mask])))}


def recurrence_predict(bits, model, order=16):
    n = bits.shape[1]-order
    result = np.full((len(bits),n), model['intercept'], dtype=np.uint8)
    for lag in model['lags']:
        offset = order-lag
        result ^= bits[:,offset:offset+n]
    return result
