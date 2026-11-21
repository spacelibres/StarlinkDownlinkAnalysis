"""Shortened-code coordinates and bounded QAM syndrome search."""
from itertools import combinations

import numpy as np

SYSTEMATIC = 4736
PARITY_START = 5120
LAST_BIT = 8192
SYMBOLS = 2731


def core_matrix(generator):
    """Rows are received bit coordinates, columns are parity checks."""
    if generator.shape != (5120, 3072):
        raise ValueError('Unexpected mother generator shape')
    return np.vstack((generator[384:], np.zeros((384, 3072), dtype=np.uint8),
                      np.eye(3072, dtype=np.uint8), np.zeros((1, 3072), dtype=np.uint8)))


def bits_from_labels(labels):
    return ((labels[:, None] >> np.array([2, 1, 0], dtype=np.uint8)) & 1).ravel()


def pack_rows(matrix):
    return [int.from_bytes(row.tobytes(), 'little') for row in np.packbits(matrix, axis=1, bitorder='little')]


def search_moves(syndrome, moves, pair_limit, max_changes=2):
    """Test all single moves and up to four moves within a fixed shortlist."""
    matches = [((s, label),) for s, label, _, delta in moves if delta == syndrome]
    closest = sorted(moves, key=lambda m: m[2])[:pair_limit]
    pairs = {}
    for i, j in combinations(range(len(closest)), 2):
        a, b = closest[i], closest[j]
        if a[0] == b[0]:
            continue
        delta = a[3] ^ b[3]
        if delta == syndrome:
            matches.append(((a[0], a[1]), (b[0], b[1])))
        if max_changes == 4:
            pairs.setdefault(delta, []).append((i, j))
    if max_changes == 2:
        return matches
    if max_changes != 4:
        raise ValueError('Search accepts at most two or four symbol changes')
    larger = set()
    for i, move in enumerate(closest):
        for j, k in pairs.get(syndrome ^ move[3], ()):
            if i < j and len({move[0], closest[j][0], closest[k][0]}) == 3:
                larger.add(tuple(sorted(closest[m][:2] for m in (i, j, k))))
    for delta, first in pairs.items():
        other = syndrome ^ delta
        if delta >= other:
            continue
        for a, b in first:
            for c, d in pairs.get(other, ()):
                pattern = tuple(sorted(closest[m][:2] for m in (a, b, c, d)))
                if len({item[0] for item in pattern}) == 4:
                    larger.add(pattern)
    return matches + sorted(larger)


def decide_word(samples, constellation, matrix, packed_columns, pair_limit, max_changes=2):
    costs = abs(samples[:, None] - constellation[None, :]) ** 2
    hard_labels = costs.argmin(axis=1).astype(np.uint8)
    hard = bits_from_labels(hard_labels)
    syndrome = (hard @ matrix) % 2
    target = pack_rows(syndrome[None, :])[0]
    if not target:
        return hard, hard.copy(), {'status': 'zero_initial_syndrome', 'initial_syndrome_weight': 0}
    moves = []
    for s, old in enumerate(hard_labels):
        a, b, c = packed_columns[3*s:3*s+3]
        deltas = [0, c, b, c ^ b, a, a ^ c, a ^ b, a ^ b ^ c]
        for label in range(8):
            delta = deltas[int(old) ^ label]
            if delta:
                moves.append((s, label, float(costs[s, label] - costs[s, old]), delta))
    solutions = {}
    for pattern in search_moves(target, moves, pair_limit, max_changes):
        changed = hard_labels.copy()
        increase = 0.
        for s, label in pattern:
            changed[s] = label
            increase += float(costs[s, label] - costs[s, hard_labels[s]])
        proposal = bits_from_labels(changed)
        proposal[SYSTEMATIC:PARITY_START] = hard[SYSTEMATIC:PARITY_START]
        proposal[LAST_BIT] = hard[LAST_BIT]
        positions = tuple(np.flatnonzero(proposal != hard).tolist())
        if positions not in solutions or increase < solutions[positions]:
            solutions[positions] = increase
    record = {'initial_syndrome_weight': int(syndrome.sum()), 'single_symbol_candidates': len(moves),
              'pair_shortlist': min(pair_limit, len(moves)), 'core_solution_count': len(solutions)}
    if len(solutions) != 1:
        record['status'] = 'ambiguous' if solutions else 'outside_search'
        return hard, hard.copy(), record
    positions, delta = next(iter(solutions.items()))
    corrected = hard.copy()
    corrected[list(positions)] ^= 1
    if ((corrected @ matrix) % 2).any():
        raise RuntimeError('Syndrome and dense generator disagree')
    record.update(status='resolved_in_search', changed_positions=list(positions), squared_distance_increase=delta)
    return hard, corrected, record
