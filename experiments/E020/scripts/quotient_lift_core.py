"""Sequential recovery in the characteristic-two coordinate t=x+1."""

import numpy as np


def pascal_transform(values):
    """Transform block coefficients between x and t=x+1 in GF(2)."""
    transformed = np.array(values, dtype=np.uint8, copy=True, order='C')
    width = transformed.shape[-1]
    if width < 1 or width & (width - 1):
        raise ValueError('Last dimension must be a power of two')
    step = 1
    while step < width:
        view = transformed.reshape(*transformed.shape[:-1], -1, 2 * step)
        view[..., :step] ^= view[..., step:]
        step *= 2
    return transformed


def fit_layers(u, v, training, word_weights=None):
    """Fit each next coefficient solely on training words."""
    dimension = u.shape[1]
    parity, depth = v.shape[1:]
    if (u.shape[0], u.shape[2]) != (len(v), depth) or dimension > 16:
        raise ValueError('Incompatible blocks or too many candidate dimensions')
    u = u.transpose(0, 2, 1)
    v = v.transpose(0, 2, 1)
    options = ((np.arange(1 << dimension)[:, None] >> np.arange(dimension)) & 1).astype(np.uint8)
    candidates = (u[training, 0] @ options.T) % 2
    signed_candidates = (1 - 2 * candidates.astype(np.int16)).T.copy()
    if word_weights is not None:
        signed_candidates = signed_candidates.astype(np.float64) * np.asarray(word_weights)[training]
    prediction = np.zeros_like(v)
    learned = np.empty((dimension, parity, depth), dtype=np.uint8)
    audits = []
    for d in range(depth):
        residual = v[training, d] ^ prediction[training, d]
        score = signed_candidates @ (1 - 2 * residual.astype(np.int16))
        best = np.argmax(score, axis=0)
        coefficient = options[best].T.copy()
        learned[:, :, d] = coefficient
        prediction[:, d:] ^= (u[:, :depth-d] @ coefficient) % 2
        maximum = score.max(axis=0)
        next_score = np.partition(score, -2, axis=0)[-2]
        audits.append({
            'degree': d,
            'selected_column_masks': best.tolist(),
            'tied_best_counts': np.count_nonzero(score == maximum, axis=0).tolist(),
            'training_score_margin': (maximum - next_score).tolist(),
            'training_agreement': np.count_nonzero(v[training, d] == prediction[training, d], axis=0).tolist(),
            'holdout_agreement': np.count_nonzero(v[~training, d] == prediction[~training, d], axis=0).tolist(),
        })
    return learned, audits, prediction.transpose(0, 2, 1)
