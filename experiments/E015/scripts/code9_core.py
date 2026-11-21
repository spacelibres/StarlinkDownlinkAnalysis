"""Whole-frame splits and systematic GF(2) checks."""
import numpy as np


def edge_slope_lifts(slopes, carriers, lower_center, upper_center):
    """Recover relative cycles along time, anchored near zero at the SSS."""
    separation = upper_center - lower_center
    wrapped = np.asarray(slopes) * separation
    phase = np.unwrap(np.r_[0.0, wrapped])[1:]
    cycles = np.rint((phase - wrapped) / (2 * np.pi)).astype(int)
    correction = (2 * np.pi / separation) * cycles[:, None] * (carriers - lower_center)
    return correction, cycles


def split_frames(frames, seed, train_groups):
    """Permute frame identifiers once, then assign every word in each frame."""
    unique = np.unique(frames)
    if not 0 < train_groups < len(unique):
        raise ValueError('Training and holdout both require whole frame groups')
    order = np.random.default_rng(seed).permutation(unique)
    return np.isin(frames, order[:train_groups]), order.tolist()


def systematic_checks(generator):
    return np.column_stack((generator.T, np.eye(generator.shape[1], dtype=np.uint8)))


def parity_disagreements(words, generator):
    dimension, parity_length = generator.shape
    if words.ndim != 2 or words.shape[1] != dimension + parity_length:
        raise ValueError('Words do not match the systematic split')
    predicted = (words[:, :dimension] @ generator) % 2
    return np.count_nonzero(predicted != words[:, dimension:], axis=1)
