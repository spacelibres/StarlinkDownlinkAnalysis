"""Symbol-likelihood scoring for projected parity equations."""

import numpy as np


def label_parity_means(probabilities):
    """E[(-1)^parity] for each of the eight three-bit masks."""
    parity_sign = np.array([[(-1)**((label & mask).bit_count()) for mask in range(8)] for label in range(8)])
    return probabilities @ parity_sign


def expected_check_agreement(means, base, toggles, options):
    """Factor exact per-symbol parity means under conditional symbol independence."""
    support = toggles != 0
    multiplicity = support.sum(axis=0)
    fixed = np.flatnonzero((multiplicity == 0) & (base != 0))
    product = np.prod(means[:, fixed, base[fixed]], axis=1)[:, None] * np.ones((1, len(options)))
    for block, selector in enumerate(options.T):
        idx = np.flatnonzero((multiplicity == 1) & support[block])
        if not len(idx):
            continue
        p0 = np.prod(means[:, idx, base[idx]], axis=1)
        p1 = np.prod(means[:, idx, base[idx] ^ toggles[block, idx]], axis=1)
        product *= np.where(selector, p1[:, None], p0[:, None])
    for i in np.flatnonzero(multiplicity > 1):
        decoded_masks = np.bitwise_xor.reduce((options * toggles[:, i]).astype(np.uint8), axis=1) ^ base[i]
        product *= means[:, i, decoded_masks]
    return product


def mask_symbols(bits):
    return (bits.reshape(-1, 3) @ np.array([4, 2, 1], dtype=np.uint8)).astype(np.uint8)
