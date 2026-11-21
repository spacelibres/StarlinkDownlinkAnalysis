"""Packed GF(2) polynomial arithmetic for power-of-two cyclic blocks."""

import numpy as np


def reduce_polynomial(value, lift):
    mask = (1 << lift) - 1
    while value >> lift:
        value = (value & mask) ^ (value >> lift)
    return value


def ring_multiply(a, b, lift):
    if a.bit_count() < b.bit_count():
        a, b = b, a
    product = 0
    while b:
        term = b & -b
        product ^= a << (term.bit_length() - 1)
        b ^= term
    return reduce_polynomial(product, lift)


def ring_inverse(value, lift):
    """Use extended Euclid against x**lift + 1."""
    if value.bit_count() % 2 == 0:
        raise ValueError('Even-weight polynomial is not a unit')
    a, b = value, (1 << lift) | 1
    u, v = 1, 0
    while a != 1:
        shift = a.bit_length() - b.bit_length()
        if shift < 0:
            a, b = b, a
            u, v = v, u
            shift = -shift
        a ^= b << shift
        u ^= v << shift
    return reduce_polynomial(u, lift)


def pack_blocks(bits, lift):
    blocks = np.asarray(bits, dtype=np.uint8).reshape(len(bits), -1, lift)
    packed = np.packbits(blocks, axis=-1, bitorder='little')
    return [[int.from_bytes(block.tobytes(), 'little') for block in row] for row in packed]


def binary_blocks(blocks, lift):
    width = (lift + 7) // 8
    raw = b''.join(b.to_bytes(width, 'little') for row in blocks for b in row)
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(len(blocks), -1, width)
    return np.unpackbits(packed, axis=-1, count=lift, bitorder='little').reshape(len(blocks), -1)


def evaluate(inputs, generator, lift):
    rows = []
    for word in inputs:
        parity = [0] * len(generator[0])
        for coefficient, row in zip(word, generator, strict=True):
            for j, value in enumerate(row):
                parity[j] ^= ring_multiply(coefficient, value, lift)
        rows.append(parity)
    return rows


def recover(inputs, parity, lift):
    """Solve U P = V with unit pivots over GF(2)[x]/(x**lift+1).

    A full set of unit pivots proves uniqueness. A missing pivot does not
    adjudicate consistency or recover a partially specified generator.
    """
    if lift < 1 or lift & (lift - 1):
        raise ValueError('Expected a power-of-two block size')
    rows = [u + v for u, v in zip(inputs, parity, strict=True)]
    dimension = len(inputs[0])
    width = len(rows[0])
    for c in range(dimension):
        pivot = next((r for r in range(c, len(rows)) if rows[r][c].bit_count() & 1), None)
        if pivot is None:
            return None, c, None
        rows[c], rows[pivot] = rows[pivot], rows[c]
        current = rows[c]
        inverse = ring_inverse(current[c], lift)
        for j in range(c, width):
            current[j] = ring_multiply(current[j], inverse, lift)
        for r, row in enumerate(rows):
            scale = row[c]
            if r == c or not scale:
                continue
            for j in range(c, width):
                row[j] ^= ring_multiply(scale, current[j], lift)
    inconsistent = sum(any(row[dimension:]) for row in rows[dimension:])
    if inconsistent:
        return None, dimension, inconsistent
    return [r[dimension:] for r in rows[:dimension]], dimension, 0


def dense_generator(generator, lift):
    first_rows = binary_blocks(generator, lift).reshape(len(generator), -1, lift)
    return np.vstack([
        np.stack([np.roll(row, shift, axis=-1).ravel() for shift in range(lift)])
        for row in first_rows
    ])


def single_bit_candidates(generator, syndrome):
    """Return every column of [P.T | I] equal to a nonzero syndrome."""
    if not np.any(syndrome):
        return []
    candidates = np.flatnonzero(np.all(generator == syndrome, axis=1)).tolist()
    nonzero = np.flatnonzero(syndrome)
    if len(nonzero) == 1:
        candidates.append(generator.shape[0] + int(nonzero[0]))
    return candidates
