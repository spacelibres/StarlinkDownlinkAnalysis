"""Withhold each observed parity block of the two soft Code 28 recoveries."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code28_corner_constraints import CacheOnly, syndrome_weights
from audit_qam_pn15 import equality_predictions, matching_run
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import pack_blocks
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, coordinates, packed_check_columns
from educational_fec_demo import encode_block, word_bits
from extract_qin_low_mcs import target_symbols
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from replay_leo_padded_code import solve_erasures
from reproduce_code28 import demap, labels_to_bits
from validate_code37 import constellation_for


def main():
    require_run_context()
    inputs = {}

    def artifact(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    corner = read(artifact('v144-code28-corner', 'E146', 'code28_corner_constraints.json'))
    earlier = read(artifact('v143-code28-archive-transfer', 'E145', 'code28_archive_transfer.json'))
    seeds = artifact('v019-code28-reproduction-repeat', 'E021', 'code28_reproduction.npz')
    with np.load(seeds, allow_pickle=False) as src:
        generator = src['generator_seed_bits'].reshape(9, 7, LIFT)
    columns = packed_check_columns([list(r) for r in pack_blocks(generator.reshape(9, -1), LIFT)], LIFT, OMITTED, 2)
    selected = {r['frame']: [(idx, trial['fraction']) for idx, w in enumerate(r['fixed_gain']['words'])
                             for trial in w['trials'] if trial['unique']] for r in corner['soft']}
    selected = {k: v for k, v in selected.items() if v}
    fixed = {r['frame']: r for r in earlier['frames']}
    references = equality_predictions(np.arange(MOTHER_N), pn_bits(), earlier['frozen_pn_model'])
    points = constellation_for(2, False, True)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    lanes = ((np.arange(8)[:, None] >> [2, 1, 0]) & 1).astype(bool).T
    reader = CacheOnly(PROJECT_ROOT/'.cache/qin-mat-pages', False)
    result, arrays = [], {}
    with reader, h5py.File(reader, 'r') as source:
        for frame, group, flat in target_symbols(source['yDataSoft'], sorted(selected), rotation, signs):
            ref = flat[228:1252]
            unfolded = ref * np.sign(ref.real)
            axis = complex(np.median(unfolded.real), np.median(unfolded.imag))
            for word_index, fraction in selected[frame]:
                start = fixed[frame]['allocation_start'] + word_index*SYMBOLS
                z = flat[start:start+SYMBOLS] / axis * max(abs(points))
                distance, labels, _ = demap(z[None, :], points)
                bits = labels_to_bits(labels).ravel()
                margin = np.stack([abs(distance[..., k].min(axis=-1)-distance[..., ~k].min(axis=-1))
                                   for k in lanes], axis=-1).ravel()
                core = coordinates(bits[None, :])[2]
                order = np.flatnonzero(core)
                order = order[np.argsort(margin[order], kind='stable')]
                keep = np.ones_like(core)
                keep[order[:round(len(order)*fraction)]] = False
                fit, rank, _ = solve_erasures(columns, bits, keep)
                if fit is None or syndrome_weights(fit[None, :], generator) != [0]:
                    raise ValueError('Frozen soft result does not independently re-encode')
                checks = []
                for block in range(7):
                    offset = MOTHER_K + block*LIFT
                    hold = np.zeros_like(keep)
                    hold[offset:offset+LIFT] = keep[offset:offset+LIFT]
                    restored, held_rank, consistent = solve_erasures(columns, bits, keep & ~hold)
                    checks.append({'block': block, 'held': int(hold.sum()), 'rank': held_rank, 'consistent': consistent,
                                   'matches': int((restored[hold] == bits[hold]).sum()) if restored is not None else None,
                                   'same_core': bool(restored is not None and np.array_equal(fit[core], restored[core]))})
                pn_before = matching_run(bits[:MOTHER_N] ^ references)
                pn_after = matching_run(fit[:MOTHER_N] ^ references)
                result.append({'frame': frame, 'group': group, 'word': word_index, 'erase_fraction': fraction,
                               'erased_core_bits': int((~keep).sum()), 'rank': rank, 'held_parity': checks,
                               'total_core_bit_changes': int((fit[core] != bits[core]).sum()),
                               'pn_before': pn_before, 'pn_after': pn_after})
                key = f'frame_{frame}_word_{word_index}_'
                arrays.update({key+'soft': z, key+'hard': bits, key+'observed_core_mask': core & keep,
                               key+'core_mask': core, key+'restored_core_values': fit[core]})
    pages = sorted(reader.receipts.values(), key=lambda p: p['bytes'][0])
    if not all(p in corner['archive_pages'] for p in pages):
        raise ValueError('Soft witness data differ from E146 scan')
    output = experiment_output('E146')
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'starlink-code28-soft-witness/v1', 'words': result, 'pages': pages,
               'inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): sha for p, sha in inputs.items()}}
    (output/'code28_soft_witnesses.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(output/'code28_soft_witnesses.npz', **arrays)
    for word in result:
        print(word['frame'], word['total_core_bit_changes'], 'held', [(r['held'], r['matches']) for r in word['held_parity']],
              'PN after', word['pn_after'], flush=True)


if __name__ == '__main__':
    main()
