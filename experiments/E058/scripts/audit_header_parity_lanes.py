"""Test convolutional headers with each third output withheld in turn."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'learning')]
from project_paths import experiment_output, require_run_context
from audit_h2_projection import SELECTION_ROOT, TRACK, WAV_MODEL, band_masks, header_bits, visible_positions
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import crc8, encode_block
from layout_core import NONPILOT
from protocol_core import GENERATORS
from receiver_models import pn_reference
from sparse_code5_core import binary_rank


HEADER_RESOURCES = 114
MAX_BLOCKS = 7
SCORE_MIN = .5
OFFSETS = (0, -114, -3, -1, 1, 3, 114)
STATES = np.arange(64)
REGISTER = (STATES[:, None] << 1) | np.arange(2)
SIGNS = np.stack([
    np.fromiter(((int(r) & g).bit_count() % 2 for r in REGISTER.ravel()), float).reshape(64, 2)
    for g in GENERATORS
], axis=-1)*2 - 1


def decode_batch(soft):
    steps = np.asarray(soft, dtype=float).reshape(-1, 38, 3)
    metric = np.full((len(steps), 64), -np.inf)
    metric[:, 0] = 0.
    history = np.empty((38, len(steps), 64), dtype=bool)
    for i in range(38):
        joint = metric[:, :, None] + np.einsum('bj,suj->bsu', steps[:, i], SIGNS)
        lo = joint[:, STATES >> 1, STATES & 1]
        hi = joint[:, (STATES >> 1)+32, STATES & 1]
        history[i] = hi > lo
        metric = np.maximum(lo, hi)
    best = np.zeros(len(steps), dtype=np.int64)
    traces = np.arange(len(steps))
    bits = np.empty((len(steps), 32), dtype=np.uint8)
    for i in range(37, -1, -1):
        if i < 32:
            bits[:, i] = best & 1
        best = (best >> 1) + 32*history[i, traces, best]
    return bits, metric[:, 0]


def encode_signs(bits):
    return np.concatenate([2*encode_block(row).astype(float)-1
                           for row in np.asarray(bits).reshape(-1, 32)])


def lane_trials(waveforms, lane, *, fixed_blocks=None):
    waveforms = np.asarray(waveforms)
    visible = np.arange(waveforms.shape[1]) % 3 != lane
    first_mask = visible[:HEADER_RESOURCES]
    first = waveforms[:, :HEADER_RESOURCES][:, first_mask]
    axis = .5*np.angle(np.sum(first**2, axis=1))
    all_soft = (waveforms*np.exp(-1j*axis[:, None])).real
    training = all_soft*visible
    first_training = training[:, :HEADER_RESOURCES]
    bits, scores = decode_batch(np.concatenate((first_training, -first_training)))
    first_index = np.arange(len(waveforms))
    sign = np.where(scores[:len(waveforms)] >= scores[len(waveforms):], 1, -1)
    chosen = bits[first_index + (sign < 0)*len(waveforms)]
    blocks = np.full(len(waveforms), fixed_blocks) if fixed_blocks is not None else chosen[:, 8:11] @ [1, 2, 4]
    decoded = chosen[:, None, :]
    available = training.shape[1]//HEADER_RESOURCES
    if available > 1:
        corrected = training[:, HEADER_RESOURCES:available*HEADER_RESOURCES]*sign[:, None]
        rest, _ = decode_batch(corrected.reshape(-1, HEADER_RESOURCES))
        decoded = np.concatenate((decoded, rest.reshape(len(waveforms), available-1, 32)), axis=1)
    first_header = chosen[:, :16]
    result = []
    for j, block_count in enumerate(blocks):
        row = {'lane': lane, 'blocks': int(block_count),
               'prefix': f"{int(''.join(map(str, first_header[j])), 2):04X}", 'bits': None,
               'crc_ok': False, 'held_coherence': None, 'held_hard_disagreements': None}
        if 2 <= block_count <= min(available, MAX_BLOCKS):
            n = int(block_count)*HEADER_RESOURCES
            observed = (all_soft[j, :n]*sign[j])[~visible[:n]]
            message = decoded[j, :block_count].reshape(-1)
            prediction = encode_signs(message)[~visible[:n]]
            energy = np.linalg.norm(observed)*np.linalg.norm(prediction)
            row.update(bits=message, crc_ok=bool(crc8(message[:-8]) == np.packbits(message[-8:])[0]),
                       held_coherence=float(np.vdot(observed, prediction).real/max(energy, 1e-30)),
                       held_hard_disagreements=int(np.count_nonzero((observed > 0) != (prediction > 0))))
        result.append(row)
    return result


def lane_trial(waveform, lane, *, fixed_blocks=None):
    return lane_trials(np.asarray(waveform)[None, :], lane, fixed_blocks=fixed_blocks)[0]


def summarize_trials(waveforms, ids, *, fixed_blocks=None, reference=None):
    groups = [lane_trials(waveforms, lane, fixed_blocks=fixed_blocks) for lane in range(3)]
    out = []
    for i, rows in enumerate(zip(*groups, strict=True)):
        recovered = [row['bits'] for row in rows]
        agreement = recovered[0] is not None and all(np.array_equal(recovered[0], b) for b in recovered[1:])
        accepted = agreement and all(row['crc_ok'] and row['held_coherence'] > SCORE_MIN for row in rows)
        out.append({
            'frame': int(ids[i]), 'same_word': agreement, 'passes_all_lanes': accepted,
            'matches_reference_by_lane': ([np.array_equal(b, reference[i]) for b in recovered]
                                          if reference is not None else None),
            'lanes': [{**row, 'bits': None if row['bits'] is None else np.packbits(row['bits']).tobytes().hex()}
                      for row in rows],
        })
    return out


def reference_grid():
    slots = np.arange(2, 302)
    return np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).ravel()


def archive_results(report, arrays):
    frames = report['frames']
    grid = reference_grid()
    starts = arrays['h2_start']
    lengths = arrays['h2_length']
    if not np.array_equal(starts, [r['h2_start'] for r in frames]) or min(lengths) != MAX_BLOCKS*HEADER_RESOURCES:
        raise ValueError('The preserved H2 ranges or corresponding frame index changed')
    ids = arrays['frame_index']
    h1 = arrays['h1_symbols'] * grid[:2*HEADER_RESOURCES].conj()
    h2 = arrays['h2_symbols'] * grid[starts[:, None]+np.arange(MAX_BLOCKS*HEADER_RESOURCES)].conj()
    return {
        name: summarize_trials(waves, ids, fixed_blocks=blocks,
                               reference=[header_bits(r[name]['words_hex']) for r in frames])
        for name, waves, blocks in (('h1', h1, 2), ('h2', h2, None))
    }


def offset_results(report, observations, first_start):
    groups = (
        ('eight', report['folds'][0]['held_ticks'], 'fold_0_held'),
        ('seven', report['folds'][1]['held_ticks'], 'fold_1_held'),
        ('overlap', report['overlap_ticks'], 'fold_0_overlap'),
    )
    result = {}
    for name, ticks, array in groups:
        vectors = observations[array]
        if len(vectors) != len(ticks):
            raise ValueError('Frame list differs from the offset observation array')
        result[name] = {}
        for offset in OFFSETS:
            start = first_start+offset
            waves = vectors[:, start:start+MAX_BLOCKS*HEADER_RESOURCES]
            result[name][str(offset)] = summarize_trials(waves, ticks)
    return result


def narrowband_ranks(inputs):
    files = [ROOT/TRACK, ROOT/WAV_MODEL, *(SELECTION_ROOT/name/'selection.json'
                                         for name in ('DSX01-S05', 'paired_RX'))]
    for path in files:
        inputs[path] = sha256(path)
    views, _ = band_masks()
    matrix = np.array([encode_signs(b) > 0 for b in np.eye(64, dtype=np.uint8)]).T.astype(np.uint8)
    results = {}
    for view, carriers in views.items():
        rows = list(visible_positions(0, 2*HEADER_RESOURCES, carriers))
        projection = matrix[rows]
        results[view] = {
            'coded_resources': rows, 'full_rank': binary_rank(projection),
            'input_dependencies': np.flatnonzero(projection.any(axis=0)).tolist(),
            'rank_after_known_16': binary_rank(projection[:, 16:]),
            'observations_per_convolutional_lane': [sum(r % 3 == j for r in rows) for j in range(3)],
        }
    return results


def main():
    require_run_context()
    inputs = {}
    archived = checked_artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json', inputs)
    waves = checked_artifact('v034-qin-header-survey', 'E036', 'qin_header_symbols.npz', inputs)
    offset = checked_artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json', inputs)
    offset_waves = checked_artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_soft.npz', inputs)
    offset_start = read(checked_artifact('v043-ut-offset-h2-final', 'E045',
                                        'ut_offset_h2_systematics.json', inputs))['h2_start_resource']
    with np.load(waves, allow_pickle=False) as f:
        archive = archive_results(read(archived), f)
    with np.load(offset_waves, allow_pickle=False) as f:
        weak = offset_results(read(offset), f, offset_start)
    ranks = narrowband_ranks(inputs)
    result = {
        'schema': 'starlink-header-parity-lanes/v1',
        'inputs_sha256': {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        'held_coherence_min': SCORE_MIN, 'offset_controls_resources': OFFSETS,
        'support': {'archive_frames': len(archive['h1']), 'offset_H2_start_resource': offset_start},
        'archive': archive, 'wideband_offset_H2': weak, 'narrowband_H1_rank': ranks,
    }
    out = experiment_output('E058')
    out.mkdir(parents=True, exist_ok=True)
    (out/'header_parity_lanes.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    for name, rows in archive.items():
        print(name, Counter(r['passes_all_lanes'] for r in rows), flush=True)
    for name, shifts in weak.items():
        print(name, {shift: sum(r['passes_all_lanes'] for r in rows) for shift, rows in shifts.items()}, flush=True)


if __name__ == '__main__':
    main()
