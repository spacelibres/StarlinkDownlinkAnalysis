"""Test Code 28 erasures that preserve quadrant information in the archive."""

from functools import reduce
import json
from operator import xor
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code28_archive_transfer import CacheOnly, LATTICE_ERROR, erase_core, syndrome_weights
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import pack_blocks
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, coordinates, packed_check_columns
from code28_receiver_core import crossfit_gain
from educational_fec_demo import encode_block, word_bits
from extract_qin_low_mcs import target_symbols
from layout_core import NONPILOT
from receiver_models import pn_reference
from replay_leo_padded_code import solve_erasures
from reproduce_code28 import demap, labels_to_bits
from validate_code37 import constellation_for


SOFT_GROUPS = (624, 637, 780)
ERASURE_FRACTIONS = (.0625, .125, .25)


def solve_domains(columns, baseline, changes):
    """Each generator of the observation domain toggles the given code coordinates."""
    effect = [reduce(xor, (columns[i] for i in indices), 0) for indices in changes]
    syndrome = reduce(xor, (columns[i] for i in np.flatnonzero(baseline)), 0)
    problem = np.zeros(len(effect)+1, dtype=np.uint8)
    problem[-1] = 1
    recovered, rank, consistent = solve_erasures(effect + [syndrome], problem, problem)
    if recovered is None:
        return None, rank, consistent
    out = baseline.copy()
    for indices, use in zip(changes, recovered[:-1], strict=True):
        if use:
            out[list(indices)] ^= 1
    return out, rank, consistent


def domain(word, clear, quadrant_used, readings, core, points, wrong_quadrant=False):
    base = word.copy()
    toggles = []
    for symbol in np.flatnonzero(clear | quadrant_used):
        start = symbol*3
        if clear[symbol]:
            toggles.extend((i,) for i in range(start, start+3) if core[i])
        else:
            x, y = readings[symbol].real > 0, readings[symbol].imag > 0
            support = np.flatnonzero(((points.real > 0) == (x ^ wrong_quadrant)) & ((points.imag > 0) == y))
            a, b = labels_to_bits(support[None, :]).reshape(2, 3)
            active = core[start:start+3]
            base[start:start+3][active] = a[active]
            delta = tuple(int(start+i) for i in np.flatnonzero((a != b) & active))
            if delta:
                toggles.append(delta)
    return base, toggles


def report_candidate(columns, word, clear, quadrant, sample, core, points, wrong=False):
    base, free = domain(word, clear, quadrant, sample, core, points, wrong)
    solution, rank, consistent = solve_domains(columns, base, free)
    return {'variables': len(free), 'rank': rank, 'consistent': consistent, 'unique': solution is not None}, solution


def hold_parity(columns, word, clear, quadrant, sample, core, points, solution):
    result = []
    measured = np.repeat(~(clear | quadrant), 3) & core
    for block in range(7):
        left = MOTHER_K + block*LIFT
        hold = np.arange(left, left+LIFT)
        affected = np.isin(np.arange(len(clear)), hold//3)
        report, fit = report_candidate(columns, word, clear | affected, quadrant & ~affected, sample, core, points)
        test = hold[measured[hold]]
        result.append({**report, 'observed': len(test),
                       'matches': int((fit[test] == word[test]).sum()) if fit is not None else None,
                       'same_core': bool(fit is not None and np.array_equal(fit[core], solution[core]))})
    return result


def audit_corner(frame, fixed, flat, points, generator, columns):
    axis = complex(*fixed['axis'])
    raw = (flat / axis).reshape(-1, len(NONPILOT))
    full_corner = np.isclose(abs(raw.real), 1/np.sqrt(2), atol=2e-6, rtol=0)
    full_corner &= np.isclose(abs(raw.imag), 1/np.sqrt(2), atol=2e-6, rtol=0)
    gates = full_corner.all(axis=1)
    start, stop = fixed['allocation_start'], fixed['allocation_stop']
    positions = np.arange(start, stop).reshape(-1, SYMBOLS)
    selected = gates[positions // len(NONPILOT)]
    out = []
    for index in np.flatnonzero(selected.any(axis=1)):
        pos = positions[index]
        sample = flat[pos]/axis*max(abs(points))
        distances, labels, _ = demap(sample[None, :], points)
        word = labels_to_bits(labels).ravel()
        core = coordinates(word[None, :])[2]
        zone = selected[index]
        edge = NONPILOT[pos % len(NONPILOT)] == 511
        bad = distances.min(axis=-1).ravel() >= LATTICE_ERROR
        baseline = bad | edge
        erased = baseline | zone
        clear = (bad & ~zone) | edge
        quadrant = zone & ~edge
        scores = {}
        for name, blank, observed, wrong in (('e145_mask', baseline, np.zeros_like(zone), False),
                                             ('erase_corner_slot', erased, np.zeros_like(zone), False),
                                             ('quad_constraints', clear, quadrant, False),
                                             ('wrong_quad_constraints', clear, quadrant, True)):
            scores[name], solution = report_candidate(columns, word, blank, observed, sample, core, points, wrong)
            if name in ('erase_corner_slot', 'quad_constraints') and solution is not None:
                if syndrome_weights(solution[None, :], generator) != [0]:
                    raise ValueError('Corner correction fails independent re-encoding')
                scores[name]['held_parity'] = hold_parity(columns, word, blank, observed, sample, core, points, solution)
        out.append({'frame': frame, 'word': int(index), 'start_resource': int(pos[0]),
                    'corner_symbols': int(zone.sum()), 'scores': scores})
    return out, np.flatnonzero(gates[start//len(NONPILOT):(stop-1)//len(NONPILOT)+1]).size


def audit_soft(flat, fixed, points, generator, columns):
    start, stop = fixed['allocation_start'], fixed['allocation_stop']
    position = np.arange(start, stop)
    ref = flat[228:1252]
    unfolded = ref * np.sign(ref.real)
    axis = complex(np.median(unfolded.real), np.median(unfolded.imag))
    z = flat[start:stop]/axis*max(abs(points))
    slots, carriers = divmod(position, len(NONPILOT))
    correction = crossfit_gain(crossfit_gain(z, slots, carriers, points), carriers, slots, points)
    output = {}
    for name, values in (('fixed_gain', z), ('crossfit', correction)):
        d, ids, _ = demap(values.reshape(-1, SYMBOLS), points)
        hard = labels_to_bits(ids)
        margins = np.stack([abs(d[..., lane].min(axis=-1)-d[..., ~lane].min(axis=-1))
                            for lane in ((np.arange(8)[:, None] >> [2, 1, 0]) & 1).astype(bool).T], axis=-1)
        core = coordinates(hard)[2]
        failed = syndrome_weights(hard, generator)
        records = []
        for w, (bits, margin) in enumerate(zip(hard, margins.reshape(hard.shape), strict=True)):
            order = np.flatnonzero(core)
            order = order[np.argsort(margin[order], kind='stable')]
            trials = []
            for fraction in ERASURE_FRACTIONS:
                keep = np.ones(len(bits), dtype=bool)
                keep[order[:round(len(order)*fraction)]] = False
                trials.append({'fraction': fraction, **erase_core(columns, bits, keep, core)})
            records.append({'failed_checks': failed[w], 'trials': trials})
        output[name] = {'point_mse': float(d.min(axis=-1).mean()), 'words': records}
    return output


def main():
    require_run_context()
    inputs = {}
    frozen = read(checked_artifact('v143-code28-archive-transfer', 'E145', 'code28_archive_transfer.json', inputs))
    entries = {r['frame']: r for r in frozen['frames']}
    saved = checked_artifact('v019-code28-reproduction-repeat', 'E021', 'code28_reproduction.npz', inputs)
    with np.load(saved, allow_pickle=False) as src:
        generator = src['generator_seed_bits'].reshape(9, 7, LIFT)
    columns = packed_check_columns([list(r) for r in pack_blocks(generator.reshape(9, -1), LIFT)], LIFT, OMITTED, 2)
    points = constellation_for(2, False, True)
    p = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    s = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    reader = CacheOnly(PROJECT_ROOT/'.cache/qin-mat-pages', False)
    corner, soft, count = [], [], 0
    with reader, h5py.File(reader, 'r') as source:
        for frame, _, flat in target_symbols(source['yDataDec'], sorted(entries), p, s):
            rows, qualified_slots = audit_corner(frame, entries[frame], flat, points, generator, columns)
            corner.extend(rows)
            count += int(qualified_slots)
        for frame, group, flat in target_symbols(source['yDataSoft'], [i for i in sorted(entries) if
                                                 entries[i]['group'] in SOFT_GROUPS], p, s):
            soft.append({'frame': frame, 'group': group, **audit_soft(flat, entries[frame], points, generator, columns)})
    baseline = {tuple(page['bytes']): page['sha256'] for page in frozen['archive']['pages']}
    pages = sorted(reader.receipts.values(), key=lambda row: row['bytes'][0])
    if any(page['sha256'] != baseline.get(tuple(page['bytes']), page['sha256']) for page in pages):
        raise ValueError('Archive page changed')
    result = {'schema': 'starlink-code28-corner-constraints/v1', 'fixed_soft_groups': SOFT_GROUPS,
              'inputs_sha256': {path.relative_to(PROJECT_ROOT).as_posix(): digest for path, digest in inputs.items()},
              'corner_slots': count, 'corner_words': corner, 'soft': soft, 'archive_pages': pages}
    out = experiment_output('E146')
    out.mkdir(parents=True, exist_ok=True)
    (out/'code28_corner_constraints.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Hard corner slots', count, 'words', len(corner), 'unique',
          {k: sum(row['scores'][k]['unique'] for row in corner) for k in corner[0]['scores']}, flush=True)
    print('Soft whole words', len(soft), 'checked under fixed gain and crossfit', flush=True)


if __name__ == '__main__':
    main()
