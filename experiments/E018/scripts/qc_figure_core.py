"""Expand published QC seed rows and invert the parity side algebraically."""

import numpy as np

from code21_ring import recover, ring_inverse, ring_multiply


def dense_checks(shifts, lift):
    checks = np.zeros((len(shifts) * lift, shifts.shape[1] * lift), dtype=np.uint8)
    rotations = np.arange(lift)
    for r, row in enumerate(shifts):
        for b, shift in enumerate(row):
            if shift >= 0:
                checks[r * lift + rotations, b * lift + (rotations + shift) % lift] = 1
    return checks


def qc_syndrome(words, shifts, lift):
    blocks = words.reshape(len(words), shifts.shape[1], lift)
    checks = []
    for row in shifts:
        terms = [np.roll(blocks[:, j, :], -int(shift), axis=1) for j, shift in enumerate(row) if shift >= 0]
        checks.append(np.bitwise_xor.reduce(terms, axis=0))
    return np.hstack(checks)


def generator_from_shifts(shifts, lift):
    """Solve H_p v = H_u u in GF(2)[x]/(x**lift+1)."""
    source_blocks = shifts.shape[1] - len(shifts)
    coefficients = [[0 if d < 0 else 1 << ((-int(d)) % lift) for d in row] for row in shifts]
    h_u = [row[:source_blocks] for row in coefficients]
    h_p = [row[source_blocks:] for row in coefficients]
    inverse_product, pivots, inconsistent = recover(h_p, h_u, lift)
    if inverse_product is None or pivots != len(shifts) or inconsistent:
        raise ValueError('Parity side does not have a full set of unit pivots')
    return [list(column) for column in zip(*inverse_product, strict=True)]


def symbol_llrs(values, points, mapping):
    """Max-log bit metrics with a fixed positive common scale."""
    cost = abs(values[:, None] - points[None, :]) ** 2
    return np.column_stack([
        cost[:, mapping[:, i] == 1].min(axis=1) - cost[:, mapping[:, i] == 0].min(axis=1)
        for i in range(mapping.shape[1])
    ]).ravel()


def one_defect_span(words, lift):
    """Find the exact binary rank when at most one ring column remains."""
    rows = [row.copy() for row in words]
    width = len(rows[0])
    units = 0
    for c in range(width):
        pivot = next((i for i in range(units, len(rows)) if rows[i][c].bit_count() & 1), None)
        if pivot is None:
            continue
        rows[units], rows[pivot] = rows[pivot], rows[units]
        inverse = ring_inverse(rows[units][c], lift)
        rows[units] = [ring_multiply(value, inverse, lift) for value in rows[units]]
        for j in range(units + 1, len(rows)):
            scale = rows[j][c]
            if not scale:
                continue
            rows[j] = [a ^ ring_multiply(b, scale, lift)
                       for a, b in zip(rows[j], rows[units], strict=True)]
        units += 1
    result = {'unit_pivots': units, 'residual_valuation': None, 'binary_rank': None}
    if units == width:
        result['binary_rank'] = units * lift
    elif units == width - 1:
        # Unit columns have vanished in the trailing rows. The ideal in
        # the single remaining column has dimension Z minus its valuation.
        common = (1 << lift) | 1
        for row in rows[units:]:
            for value in row:
                a, b = common, value
                while b:
                    while a.bit_length() >= b.bit_length():
                        a ^= b << (a.bit_length() - b.bit_length())
                    a, b = b, a
                common = a
        valuation = common.bit_length() - 1
        result.update(residual_valuation=valuation, binary_rank=width * lift - valuation)
    return result
