"""Coordinate map and packed syndrome operations for the Code 28 core."""

import numpy as np

from code37_core import bits_from_labels, search_moves

SYSTEMATIC = 8960
MOTHER_K = 9216
MOTHER_N = 16384
SYMBOLS = 5462
OMITTED = 256
LIFT = 1024


def coordinates(observed):
    if observed.ndim != 2 or observed.shape[1] != 3 * SYMBOLS:
        raise ValueError('Expected all three-bit labels including the two spare positions')
    u = np.column_stack((np.zeros((len(observed), OMITTED), dtype=np.uint8), observed[:, :SYSTEMATIC]))
    mask = np.ones(observed.shape[1], dtype=bool)
    mask[SYSTEMATIC:MOTHER_K] = False
    mask[MOTHER_N:] = False
    return u, observed[:, MOTHER_K:MOTHER_N], mask


def packed_check_columns(generator, lift, omitted, spare):
    """Each integer holds the full parity response to one observed bit."""
    low = (1 << lift) - 1
    response = []
    for polynomials in generator:
        row = polynomials.copy()
        for _ in range(lift):
            response.append(sum(v << (i * lift) for i, v in enumerate(row)))
            row = [((v << 1) & low) | (v >> (lift - 1)) for v in row]
    parity_count = len(generator[0]) * lift
    return response[omitted:] + [0] * omitted + [1 << i for i in range(parity_count)] + [0] * spare


def repair_one(hard_labels, squared_distances, full_syndrome, columns, mask, shortlist=256):
    """Preserve untested bits and accept a unique bounded response."""
    target = int.from_bytes(np.packbits(full_syndrome, bitorder='little').tobytes(), 'little')
    if not target:
        return [], {'status': 'initially_valid', 'candidate_core_solutions': 1}
    moves = []
    for i, label in enumerate(hard_labels):
        a, b, c = columns[3 * i:3 * i + 3]
        xors = [0, c, b, c ^ b, a, a ^ c, a ^ b, a ^ b ^ c]
        current = squared_distances[i, label]
        for other in range(8):
            delta = xors[int(label) ^ other]
            if delta:
                moves.append((i, other, float(squared_distances[i, other] - current), delta))
    observed = bits_from_labels(hard_labels)
    solutions = {}
    for pattern in search_moves(target, moves, shortlist, max_changes=4):
        revised = hard_labels.copy()
        for i, label in pattern:
            revised[i] = label
        change = tuple(np.flatnonzero((bits_from_labels(revised) != observed) & mask).tolist())
        cost = sum(float(squared_distances[i, label] - squared_distances[i, hard_labels[i]]) for i, label in pattern)
        solutions[change] = min(cost, solutions.get(change, np.inf))
    record = {'status': 'ambiguous' if len(solutions) > 1 else 'outside_search',
              'candidate_core_solutions': len(solutions), 'shortlist': min(shortlist, len(moves))}
    if len(solutions) == 1:
        change, cost = next(iter(solutions.items()))
        record.update(status='resolved_in_search', squared_distance_increment=cost)
        return list(change), record
    return [], record
