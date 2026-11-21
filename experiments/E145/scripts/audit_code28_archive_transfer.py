"""Test frozen Code 28 shortening and the system/gap boundary on Qin's capture."""

from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code28_reencoding import fft_binary_parities
from audit_full_tail_phase import CacheOnly
from audit_qam_pn15 import equality_predictions, matching_run
from audit_qin_lengths import header_geometry, regions_for
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import pack_blocks
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, SYSTEMATIC, coordinates, packed_check_columns
from educational_fec_demo import encode_block, word_bits
from extract_qin_low_mcs import target_symbols
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from replay_leo_padded_code import solve_erasures
from reproduce_code28 import demap, labels_to_bits
from snapshot_qin_exemplars import ETAG, SIZE, URL
from validate_code37 import constellation_for


LATTICE_ERROR = 1e-10
NEAR_EDGE_BITS = 8


def syndrome_weights(bits, generator):
    first, parity, _ = coordinates(bits)
    estimate, _ = fft_binary_parities(first.reshape(-1, 9, LIFT), generator)
    return (estimate ^ parity).sum(axis=1).tolist()


def erase_core(columns, bits, observed, core):
    keep = observed | ~core
    restored, rank, consistent = solve_erasures(columns, bits, keep)
    return {'erased_core_bits': int((~keep).sum()), 'rank': rank, 'consistent': consistent,
            'unique': restored is not None}


def replay_allocation(samples, start, generator, columns, axis, references):
    samples = samples.reshape(-1, SYMBOLS)
    points = constellation_for(2, False, True)
    normal = samples / axis * max(abs(points))
    distances, labels, _ = demap(normal, points)
    bits = labels_to_bits(labels)
    opposite = labels_to_bits(demap(-normal, points)[1])
    core = coordinates(bits)[2]
    lattice = np.repeat(distances.min(axis=-1) < LATTICE_ERROR, 3, axis=-1)
    carrier = NONPILOT[(start + np.arange(samples.size)) % len(NONPILOT)].reshape(samples.shape)
    edge = np.repeat(carrier == 511, 3, axis=-1)
    hard = syndrome_weights(bits, generator)
    inverted = syndrome_weights(opposite, generator)
    rows = []
    for index, word in enumerate(bits):
        padding, shifted = [matching_run(word[:MOTHER_N] ^ reference) for reference in references]
        rows.append({'word': index, 'start_resource': start + SYMBOLS*index, 'last': index == len(bits)-1,
                     'hard_failed_checks': hard[index], 'inverted_failed_checks': inverted[index],
                     'lattice_erasures': erase_core(columns, word, lattice[index], core),
                     'lattice_plus_511_erasures': erase_core(columns, word, lattice[index] & ~edge[index], core),
                     'padding': padding, 'shifted_pn': shifted,
                     'gap_ones': int(word[SYSTEMATIC:MOTHER_K].sum())})
    return rows


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    generator_path = artifact('v019-code28-reproduction-repeat', 'E021', 'code28_reproduction.npz')
    with np.load(generator_path, allow_pickle=False) as old:
        generator = old['generator_seed_bits'].reshape(9, 7, LIFT)
        corrected = old['final_bits']
    columns = packed_check_columns([list(row) for row in pack_blocks(generator.reshape(9, -1), LIFT)], LIFT, OMITTED, 2)
    positive_weights = syndrome_weights(corrected, generator)
    if any(positive_weights):
        raise ValueError('Legacy Code 28 failed the frozen generator control')
    retained = np.arange(corrected.shape[1]) % 29 != 0
    core = coordinates(corrected[:1])[2]
    positive = [erase_core(columns, word, retained, core) for word in corrected]

    survey = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))
    heads = {r['frame']: r for r in survey['frames'] if 28 in header_geometry(r)[1]}
    tail = read(artifact('v065-full-tail-phase-final', 'E067', 'tail_phase_audit.json'))
    widths = {int(k): v for k, v in tail['resource_widths'].items()}
    independent_end = {r['frame']: r['comparison']['refined_minus_separate'] for r in tail['qin']}
    frozen = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))['frozen_pn_model']
    refs = [equality_predictions(np.arange(MOTHER_N), pn_bits(), {**frozen, 'phase': frozen['phase']+shift})
            for shift in (0, 1)]
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    reader = CacheOnly(PROJECT_ROOT / '.cache/qin-mat-pages', False)
    rows = []
    with reader, h5py.File(reader, 'r') as f:
        for frame, group, flat in target_symbols(f['yDataDec'], sorted(heads), rotation, signs):
            data = flat[228:228+1024]
            signed = data * np.sign(data.real)
            axis = complex(np.median(signed.real), np.median(signed.imag))
            if not np.isfinite(axis) or axis.real <= 0:
                raise ValueError('First Code 5 phase reference failed')
            _, count, start, stop = next(r for r in regions_for(heads[frame], widths) if r[0] == 28)
            rows.append({'frame': frame, 'group': group, 'axis': [axis.real, axis.imag],
                         'count': count, 'allocation_start': start, 'allocation_stop': stop,
                         'tail_residual_resources': independent_end[frame],
                         'words': replay_allocation(flat[start:stop], start, generator, columns, axis, refs)})
    pages = list(reader.receipts.values())
    known_pages = {tuple(page['bytes']): page['sha256'] for page in tail['source_qin']['pages']}
    if any(known_pages[tuple(page['bytes'])] != page['sha256'] for page in pages):
        raise ValueError('Qin pages differ from the frozen E067 archive')

    padding = {}
    for field in ('padding', 'shifted_pn'):
        padding[field] = dict(Counter((int(word['last']), int(abs(word[field]['stop']-SYSTEMATIC) <= NEAR_EDGE_BITS))
                                     for r in rows for word in r['words'] if word[field]['accepted']))
    out = experiment_output('E145')
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'starlink-code28-archive-transfer/v1', 'archive': {'url': URL, 'etag': ETAG, 'bytes': SIZE,
              'pages': sorted(pages, key=lambda page: page['bytes'][0])},
              'inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): v for p, v in inputs.items()},
              'lattice_squared_error': LATTICE_ERROR, 'legacy_positive_with_every_29th_bit_erased': positive,
              'frames': rows, 'frozen_pn_model': frozen, 'padding_stop_tolerance_bits': NEAR_EDGE_BITS}
    (out/'code28_archive_transfer.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('Legacy exact', len(positive_weights), 'legacy unique masked', sum(r['unique'] for r in positive), flush=True)
    print('Qin', len(rows), 'frames', sum(r['count'] for r in rows), 'words. Accepted PN by (last, near edge):', padding, flush=True)


if __name__ == '__main__':
    main()
