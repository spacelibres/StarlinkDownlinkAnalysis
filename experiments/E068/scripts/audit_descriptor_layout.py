"""Test archived descriptor layouts with independently measured terminal residues."""

from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'learning')]
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import crc8
from layout_core import little, parse_descriptors, prefix_fields


SEED = 2026112168
MODULI = (3, 4, 5)
CRT = (40, 45, 36)


def modular_system(matrix, target, modulus):
    a = np.asarray(matrix, dtype=np.int64).copy() % modulus
    b = np.asarray(target, dtype=np.int64).copy() % modulus
    transform = np.eye(len(a), dtype=np.int64)
    pivots = []
    for column in range(a.shape[1]):
        r = len(pivots)
        values = a[r:, column]
        eligible = np.flatnonzero(values % 2 if modulus == 4 else values)
        if len(eligible) == 0:
            continue
        pivot = r+eligible[0]
        inverse = pow(int(a[pivot, column]), -1, modulus)
        for data in (a, b, transform):
            data[[r, pivot]] = data[[pivot, r]]
            data[r] = (data[r]*inverse) % modulus
        coefficients = a[:, column].copy()
        coefficients[r] = 0
        a = (a-coefficients[:, None]*a[r]) % modulus
        b = (b-coefficients*b[r]) % modulus
        transform = (transform-coefficients[:, None]*transform[r]) % modulus
        pivots.append(column)
    r = len(pivots)
    free = np.setdiff1d(np.arange(a.shape[1]), pivots)
    remaining = b[r:]
    if modulus == 4:
        odd = np.flatnonzero(remaining % 2)
        if len(odd):
            return {'witness': {'modulus': 2, 'coefficients': transform[r+odd[0]] % 2}}
        secondary = modular_system(a[r:][:, free]//2, remaining//2, 2)
        if 'witness' in secondary:
            combined = secondary['witness']['coefficients'] @ transform[r:]
            return {'witness': {'modulus': 4, 'coefficients': combined % 4}}
        free_point = secondary['particular']
        free_directions = np.vstack((secondary['homogeneous'], 2*np.eye(len(free), dtype=np.int64)))
    else:
        nonzero = np.flatnonzero(remaining)
        if len(nonzero):
            return {'witness': {'modulus': modulus, 'coefficients': transform[r+nonzero[0]]}}
        free_point = np.zeros(len(free), dtype=np.int64)
        free_directions = np.eye(len(free), dtype=np.int64)
    particular = np.zeros(a.shape[1], dtype=np.int64)
    particular[free] = free_point
    particular[pivots] = (b[:r]-a[:r, free] @ free_point) % modulus
    directions = np.zeros((len(free_directions), a.shape[1]), dtype=np.int64)
    directions[:, free] = free_directions
    directions[:, pivots] = (-free_directions @ a[:r, free].T) % modulus
    return {'particular': particular, 'homogeneous': directions, 'unit_pivots': r}


def predict_residues(fits, proposed):
    fixed = np.ones(len(proposed), dtype=bool)
    value = np.zeros(len(proposed), dtype=np.int64)
    for fit, m, weight in zip(fits, MODULI, CRT, strict=True):
        variable = np.any((proposed @ fit['homogeneous'].T) % m != 0, axis=1)
        fixed &= ~variable
        value += weight*((proposed @ fit['particular']) % m)
    return value % 60, fixed


def bits_from_record(record):
    packed = b''.join(int(v, 16).to_bytes(4, 'big') for v in record['words_hex'])
    data = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    actual = int.from_bytes(packed[-1:], 'big')
    if len(data) != 32*record['blocks'] or not record['crc_ok'] or crc8(data[:-8]) != actual:
        raise ValueError('Frozen header CRC or dimensions failed replay')
    return data


def extract(bits, width, label_size, order):
    count = prefix_fields(bits)['entry_count']
    stop = 16+count*width
    if stop > len(bits)-8 or bits[stop:-8].any():
        return None
    entries = bits[16:stop].reshape(count, width)
    parsed = [(little(row[:label_size]), little(row[label_size:] if order == 'little' else row[label_size:][::-1]))
              for row in entries]
    return parsed if all(q > 0 for _, q in parsed) else None


def certificate(fit, matrix, target, frames):
    proof = fit['witness']
    coefficients, m = proof['coefficients'], proof['modulus']
    right = int(coefficients @ target % m)
    if np.any(coefficients @ matrix % m) or right == 0:
        raise ValueError('Inconsistent modular certificate')
    return {'modulus': m, 'rhs_residue': right, 'equations': [[int(f), int(c)] for f, c in zip(frames, coefficients)
                                                             if c]}


def fit_absolute(matrix, head, coarse, mask, residues):
    chosen = mask & (coarse >= 0)
    known = matrix[chosen]
    delta = coarse[chosen]-head[chosen]-known @ residues
    estimate, _, rank, _ = np.linalg.lstsq(known, delta/60, rcond=None)
    if rank != matrix.shape[1]:
        return None
    units = residues+60*np.rint(estimate).astype(np.int64)
    return {'units': units.tolist(), 'coarse_training_rows': int(chosen.sum()),
            'coarse_training_rank': int(rank)}


def evaluate(layout, entries, frames, groups, training_groups, head, measured, coarse, refined):
    admitted = np.array([x is not None for x in entries])
    codes = sorted({label for row in entries if row is not None for label, _ in row})
    column = {k: i for i, k in enumerate(codes)}
    design = np.zeros((len(frames), len(codes)), dtype=np.int64)
    for i, row in enumerate(entries):
        for label, count in row or ():
            design[i, column[label]] += count
    gate = admitted & (measured >= 0)
    partition = np.isin(groups, training_groups)
    train, held = gate & partition, gate & ~partition
    a, y = design[train], (measured-head)[train]
    out = {'layout': layout, 'labels': codes, 'packing_and_positive': int(admitted.sum()),
           'rejected_frames': frames[~admitted].tolist(),
           'training_frames': frames[train].tolist(), 'held_candidate_frames': frames[held].tolist()}
    fits = [modular_system(a, y, m) for m in MODULI]
    out['inconsistency'] = [certificate(fit, a, y, frames[train]) for fit in fits if 'witness' in fit]
    if out['inconsistency']:
        return out
    predict, fixed = predict_residues(fits, design[held])
    units, identified = predict_residues(fits, np.eye(len(codes), dtype=np.int64))
    residual = (predict+head[held]-measured[held]) % 60
    out['fixed_unit_residues'] = {str(k): int(v) for k, v, yes in zip(codes, units, identified) if yes}
    out['held_phase'] = {
        'fixed': int(fixed.sum()), 'matched': int(np.count_nonzero(fixed & (residual == 0))),
        'erasures': frames[held][~fixed].tolist(),
        'predictions': [[int(f), int((y0+h) % 60), int(actual)]
                        for f, y0, h, actual, valid in zip(frames[held], predict, head[held], measured[held], fixed)
                        if valid]}
    if identified.all():
        absolute = fit_absolute(design, head, coarse, train, units)
        if absolute is not None:
            endpoints = head+design @ np.asarray(absolute['units'])
            absolute['held_endpoints'] = [[int(f), int(ex), int(c), int(r)]
                                          for f, ex, c, r in zip(frames[held], endpoints[held], coarse[held], refined[held])]
            out['absolute'] = absolute
    return out


def alternate_groups(groups, measured, fixed_groups, design, head, rng):
    eligible = [g for g in fixed_groups if np.count_nonzero((groups == g) & (measured >= 0)) == 13]
    masks = [np.flatnonzero(groups == g) for g in eligible]
    positions = np.concatenate(masks)
    observations = []
    for _ in range(32):
        order = rng.permutation(len(masks))
        shuffled = np.concatenate([measured[masks[i]] for i in order])
        fit = modular_system(design[positions], shuffled-head[positions], 5)
        observations.append({'source_group_order': [int(eligible[i]) for i in order],
                             'inconsistent_mod_5': 'witness' in fit})
    return [int(g) for g in eligible], observations


def main():
    require_run_context()
    inputs = {}
    def archive(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))
    headers = archive('v034-qin-header-survey', 'E036', 'qin_header_survey.json')['frames']
    tails = archive('v065-full-tail-phase-final', 'E067', 'tail_phase_audit.json')
    h1, h2 = [[bits_from_record(row[key]) for row in headers] for key in ('h1', 'h2')]
    frames = np.array([r['frame'] for r in tails['qin']])
    if frames.tolist() != [x['frame'] for x in headers]:
        raise ValueError('Archive record IDs disagree')
    groups = np.array([r['group'] for r in tails['qin']])
    head = np.array([h['h2_start']+114*h['h2']['blocks'] for h in headers])
    signal = [r['signal'] for r in tails['qin']]
    short = [r.get('secondary_short', {}) for r in tails['qin']]
    measured = np.array([s['periodic_phase'] if s['kind'] == 'periodic' else v.get('phase', -1)
                         if v.get('accepted', False) else -1 for s, v in zip(signal, short, strict=True)])
    coarse, refined = [np.array([s[key] if s[key] is not None else -1 for s in signal])
                       for key in ('coarse_start', 'refined_start')]
    rng = np.random.default_rng(SEED)
    group_order = rng.permutation(np.unique(groups))
    train = np.sort(group_order[:len(group_order)//2])
    field_widths = {}
    for width in range(18, 25):
        possible = [16+prefix_fields(b)['entry_count']*width <= len(b)-8 for b in h2]
        admitted = [ok and not np.any(b[16+prefix_fields(b)['entry_count']*width:-8])
                    for b, ok in zip(h2, possible, strict=True)]
        field_widths[str(width)] = {'fits_capacity': sum(possible), 'zero_residual': int(np.count_nonzero(admitted))}
    candidates = []
    for width, bits, order in product((19, 20), range(6, 10), ('little', 'big')):
        key = {'width': width, 'label_bits': bits, 'quantity_order': order}
        rows = [extract(b, width, bits, order) for b in h2]
        candidates.append(evaluate(key, rows, frames, groups, train, head, measured, coarse, refined))
    h1_equivalence = []
    for row in h1:
        narrow, wide = [parse_descriptors(row, w) for w in (19, 20)]
        if not (narrow['packing_valid'] and wide['packing_valid']):
            raise ValueError('Unexpected H1 padding')
        a, b = (model['entries'][0] for model in (narrow, wide))
        h1_equivalence.append(a['mcs_candidate'] == b['mcs_candidate'] == 5 and
                              a['quantity']*1024 == b['quantity']*2048)
    expected = {'width': 20, 'label_bits': 8, 'quantity_order': 'little'}
    exact_rows = [extract(b, 20, 8, 'little') for b in h2]
    known_types = sorted({k for row in exact_rows for k, _ in row})
    design = np.array([[sum(n for k, n in row if k == c) for c in known_types] for row in exact_rows])
    controls_groups, controls = alternate_groups(groups, measured, train, design, head, rng)
    for item in candidates:
        print(item['layout'], item['packing_and_positive'], 'certificates', len(item['inconsistency']),
              'held predictions', item.get('held_phase', {}).get('matched'), flush=True)
    out = experiment_output('E068')
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'descriptor-layout-identifiability/v1', 'seed': SEED,
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
              'training_groups': train.tolist(), 'test_groups': sorted(set(groups.tolist())-set(train.tolist())),
              'field_widths': field_widths, 'h1_scaled_equivalence': int(sum(h1_equivalence)),
              'historical_h2_layout': expected, 'candidates': candidates,
              'phase_controls_training_groups': controls_groups, 'phase_controls': controls}
    (out/'descriptor_layout_identifiability.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
