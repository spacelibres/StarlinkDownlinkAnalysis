"""Withhold observed parity blocks of the unique Qin Code 28 transfer witness."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code28_archive_transfer import CacheOnly, LATTICE_ERROR, syndrome_weights
from audit_code28_reencoding import fft_binary_parities
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import pack_blocks
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, SYSTEMATIC, coordinates, packed_check_columns
from educational_fec_demo import encode_block, word_bits
from extract_qin_low_mcs import target_symbols
from layout_core import NONPILOT
from receiver_models import pn_reference
from replay_leo_padded_code import solve_erasures
from reproduce_code28 import demap, labels_to_bits
from validate_code37 import constellation_for


def main():
    require_run_context()
    inputs = {}
    transfer = read(checked_artifact('v143-code28-archive-transfer', 'E145', 'code28_archive_transfer.json', inputs))
    [(source, word)] = [(row, word) for row in transfer['frames'] for word in row['words']
                        if word['lattice_plus_511_erasures']['unique']]
    generator_path = checked_artifact('v019-code28-reproduction-repeat', 'E021', 'code28_reproduction.npz', inputs)
    with np.load(generator_path, allow_pickle=False) as old:
        generator = old['generator_seed_bits'].reshape(9, 7, LIFT)
    columns = packed_check_columns([list(row) for row in pack_blocks(generator.reshape(9, -1), LIFT)], LIFT, OMITTED, 2)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    reader = CacheOnly(PROJECT_ROOT / '.cache/qin-mat-pages', False)
    with reader, h5py.File(reader, 'r') as f:
        [(_, _, flat)] = target_symbols(f['yDataDec'], [source['frame']], rotation, signs)
    pages = sorted(reader.receipts.values(), key=lambda p: p['bytes'][0])
    if not all(page in transfer['archive']['pages'] for page in pages):
        raise ValueError('Witness pages differ from E145 transfer')
    start = word['start_resource']
    samples = flat[start:start+SYMBOLS]
    points = constellation_for(2, False, True)
    distances, labels, _ = demap((samples / complex(*source['axis']) * max(abs(points)))[None, :], points)
    bits = labels_to_bits(labels).ravel()
    core = coordinates(bits[None, :])[2]
    lattice = np.repeat(distances.min(axis=-1).ravel() < LATTICE_ERROR, 3)
    edge = np.repeat(NONPILOT[(start+np.arange(SYMBOLS)) % len(NONPILOT)] == 511, 3)
    known = (lattice & ~edge) | ~core
    full, _, _ = solve_erasures(columns, bits, known)
    if full is None or syndrome_weights(full[None, :], generator) != [0]:
        raise ValueError('Witness no longer independently re-encodes')
    results = []
    for block in range(7):
        left = MOTHER_K + block*LIFT
        hold = known.copy()
        hold[:left] = hold[left+LIFT:] = False
        solution, rank, consistent = solve_erasures(columns, bits, known & ~hold)
        results.append({'parity_block': block, 'held_observations': int(hold.sum()), 'rank': rank,
                        'consistent': consistent, 'unique': solution is not None,
                        'matches_held': int((solution[hold] == bits[hold]).sum()) if solution is not None else None,
                        'same_restoration': bool(np.array_equal(solution, full))})
    alternate = np.r_[bits[SYSTEMATIC:MOTHER_K].astype(bool), full[:SYSTEMATIC]]
    estimate, _ = fft_binary_parities(alternate.reshape(1, 9, LIFT), generator)
    gap_substitution_errors = np.count_nonzero((estimate[0] != bits[MOTHER_K:MOTHER_N]) & known[MOTHER_K:MOTHER_N])
    out = experiment_output('E145')
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'starlink-code28-archive-witness/v1', 'frame': source['frame'], 'word': word['word'],
              'allocation_start': source['allocation_start'], 'word_start': start, 'anchor': source['axis'],
              'inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'archive_pages': pages, 'held_parity': results, 'known_parity_bits': int(known[MOTHER_K:MOTHER_N].sum()),
              'gap_substitution_mismatches': int(gap_substitution_errors), 'core_erasure_coordinates': np.flatnonzero(~known).tolist()}
    (out / 'code28_witness.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(out / 'code28_witness.npz', received_symbols=samples, received_labels=bits,
                        measured_core_mask=known & core, core_mask=core, restored_core_values=full[core])
    print('Frame', source['frame'], 'word', word['word'], 'held parity', results, flush=True)


if __name__ == '__main__':
    main()
