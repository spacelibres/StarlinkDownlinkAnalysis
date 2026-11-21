"""Test structure-constrained H2 decoding on the recorded offset time grid."""

from functools import cache
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_offset_h2_systematics import BYTE_BANK, orient_h1
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import crc8, encode_block, word_bits
from layout_core import parse_descriptors, prefix_fields
from protocol_core import GENERATORS


SEED = 20261121109
CONTROLS = 63
TRIPLET_FOLDS = 8
WIDTH = 20
PREFIX_LEN = 16
STATES = np.arange(64)
INPUT = STATES & 1
REGISTER = (STATES[:, None] << 1) | np.arange(2)
SIGNS = np.empty((64, 2, 3), dtype=float)
for index, generator in enumerate(GENERATORS):
    parity = [[(int(value) & generator).bit_count() % 2 for value in pair] for pair in REGISTER]
    SIGNS[:, :, index] = 1 - 2 * np.asarray(parity)


def constrained_block(soft, fixed):
    """Return one optimum terminated word per row of -1/0/1 info-bit constraints."""
    if soft.shape != (114,) or fixed.ndim != 2 or fixed.shape[1] != 32:
        raise ValueError('Expected a 114-value observation and 32 information constraints per row')
    current = np.full((len(fixed), 64), -np.inf)
    current[:, 0] = 0
    choice = np.zeros((38, len(fixed), 64), dtype=bool)
    contributions = np.einsum('suj,tj->tsu', SIGNS, soft.reshape(38, 3))
    known = np.concatenate((fixed, np.zeros((len(fixed), 6), dtype=fixed.dtype)), axis=1)
    candidates = (known[:, :, None] < 0) | (known[:, :, None] == INPUT[None, None, :])
    for t, branch in enumerate(contributions):
        forward = current[:, :, None] + branch
        left = forward[:, STATES >> 1, INPUT]
        right = forward[:, (STATES >> 1) + 32, INPUT]
        choice[t] = right > left
        current = np.where(candidates[:, t], np.maximum(left, right), -np.inf)
    best = current[:, 0]
    track = np.zeros(len(fixed), dtype=int)
    index = np.arange(len(fixed))
    decoded = np.zeros((len(fixed), 32), dtype=np.uint8)
    for t in range(37, -1, -1):
        if t < 32:
            decoded[:, t] = track & 1
        track = (track >> 1) + 32 * choice[t, index, track]
    if np.any(track) or not np.isfinite(best).all():
        raise ValueError('Invalid constrained trellis')
    return decoded, best


@cache
def legal_prefixes(max_blocks):
    constraints = []
    for suffix in range(256):
        prefix = word_bits((0x3400 + suffix) << 16)
        fields = prefix_fields(prefix)
        n, count = fields['blocks'], fields['entry_count']
        if 2 <= n <= max_blocks and 1 <= count and PREFIX_LEN + WIDTH * count + 8 <= 32 * n:
            constraints.append(np.r_[prefix[:16].astype(np.int8), np.full(16, -1, dtype=np.int8)])
    return np.asarray(constraints)


def decode_structured(soft):
    fixed = legal_prefixes(min(7, len(soft) // 114))
    if not len(fixed):
        raise ValueError('Two complete encoded blocks are required for H2')
    first, scores = constrained_block(soft[:114].real, fixed)
    bits = first[scores.argmax()]
    prefix = prefix_fields(bits)
    blocks, count = prefix['blocks'], prefix['entry_count']
    mask = np.full(32 * blocks, -1, dtype=np.int8)
    mask[PREFIX_LEN + count * WIDTH:-8] = 0
    words = [bits]
    for block in range(1, blocks):
        data, _ = constrained_block(soft[block * 114:(block + 1) * 114].real, mask[None, block * 32:(block + 1) * 32])
        words.append(data[0])
    return np.concatenate(words)


def coded_bits(bits):
    return np.concatenate([encode_block(word) for word in bits.reshape(-1, 32)])


def cosine(values, signs):
    return float((values @ signs) / max(np.linalg.norm(values) * np.linalg.norm(signs), 1e-30))


def evaluate(bits, soft):
    code = coded_bits(bits)
    observed_crc = int(''.join(map(str, bits[-8:])), 2)
    layout = parse_descriptors(bits, WIDTH)
    return {'crc_ok': observed_crc == crc8(bits[:-8]), 'observed_crc': observed_crc,
            'words_hex': [f'0x{int("".join(map(str, word)), 2):08X}' for word in bits.reshape(-1, 32)],
            'layout': layout, 'coding_length': len(code),
            'coded_cosine': cosine(soft[:len(code)].real, 1 - 2 * code.astype(float))}


def keep_after_prefix(bits):
    """Encoded positions whose current input is a descriptor bit, even across block ends."""
    n = len(bits) // 32
    end = PREFIX_LEN + WIDTH * prefix_fields(bits)['entry_count']
    bit_positions = np.arange(32 * n).reshape(n, 32)
    coded_positions = np.repeat(np.pad(bit_positions, ((0, 0), (0, 6)), constant_values=-1), 3).ravel()
    return (coded_positions >= PREFIX_LEN) & (coded_positions < end)


def held_header(soft, erased_mask, bits):
    erased = soft.copy()
    erased[erased_mask] = 0
    alternative = decode_structured(erased)
    signs = 1 - 2 * coded_bits(alternative).astype(float)
    use = keep_after_prefix(alternative) & erased_mask[:len(signs)]
    row = {'same_retained_bits': np.array_equal(alternative, bits),
           'descriptor_resources': int(use.sum()), 'held_cosine': cosine(soft[:len(signs)].real[use], signs[use]),
           **evaluate(alternative, erased)}
    return row, alternative


def audit_header(soft, triplets):
    bits = decode_structured(soft)
    row = evaluate(bits, soft)
    branch = np.arange(len(soft)) % 3
    held = []
    for generator in range(3):
        result, _ = held_header(soft, branch == generator, bits)
        held.append({'generator_octal': f'{GENERATORS[generator]:o}', **result})
    folds = []
    matched = np.ones_like(bits, dtype=bool)
    compatible = 0
    for fold in range(TRIPLET_FOLDS):
        result, alternative = held_header(soft, triplets[:len(soft)] == fold, bits)
        if np.array_equal(alternative[:PREFIX_LEN], bits[:PREFIX_LEN]):
            matched &= alternative == bits
            compatible += 1
        folds.append({'fold': fold, **result})
    ending = PREFIX_LEN + WIDTH * prefix_fields(bits)['entry_count']
    return {**row, 'generator_holdouts': held, 'triplet_holdouts': folds, 'same_prefix_triplet_folds': compatible,
            'stable_descriptor_bits': np.flatnonzero(matched[PREFIX_LEN:ending]).tolist() if compatible == len(folds) else []}


def local_prefix_phase(soft):
    gain = soft[:24] @ BYTE_BANK[0x34]
    return soft * (gain.conjugate() / max(abs(gain), 1e-30))


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    older = read(artifact('v043-ut-offset-h2-final', 'E045', 'ut_offset_h2_systematics.json'))
    offset_run = 'v042-ut-offset-grid-header'
    receiver = read(artifact(offset_run, 'E044', 'ut_offset_grid_pilots.json'))
    weak = artifact(offset_run, 'E044', 'ut_offset_grid_soft.npz')
    comparison = 'v008-complete-header-layout-02'
    control = read(artifact(comparison, 'E010', 'full_header_layout.json'))
    groups = {'eight': 'fold_0_held', 'seven': 'fold_1_held', 'overlap': 'fold_0_overlap'}
    start = older['h2_start_resource']
    rng = np.random.default_rng(SEED)
    shuffle = np.array([rng.permutation(250) for _ in range(CONTROLS)])
    triplets = np.r_[np.full(48, -1), np.repeat(rng.permutation(250) % TRIPLET_FOLDS, 3)]
    rows = []
    with np.load(weak, allow_pickle=False) as source:
        alternative = {}
        for fold, subset in ((0, 'train'), (1, 'train'), (1, 'overlap')):
            ticks = receiver['overlap_ticks'] if subset == 'overlap' else receiver['folds'][fold]['train_ticks']
            traces, _ = orient_h1(source[f'fold_{fold}_{subset}'], older['h1_words_hex'])
            alternative.update(zip(ticks, traces, strict=True))
        for group, key in groups.items():
            aligned, _ = orient_h1(source[key], older['h1_words_hex'])
            for tick, vector in zip(older['tick_groups'][group], aligned, strict=True):
                original = vector[start:start + 798]
                shifted = {}
                for offset in (-114, -1, 1, 114):
                    displaced = vector[start + offset:start + offset + 798]
                    shifted[str(offset)] = {name: evaluate(decode_structured(trace), trace)['crc_ok']
                                            for name, trace in (('h1_phase', displaced),
                                                                ('local_phase', local_prefix_phase(displaced)))}
                local = local_prefix_phase(original)
                other = local_prefix_phase(alternative[tick][start:start + 798])
                row = {'group': group, 'tick': tick, 'known_byte_cosine': cosine(original[:24].real, BYTE_BANK[0x34]),
                       'h1_phase': audit_header(original, triplets), 'local_phase': audit_header(local, triplets),
                       'other_pilot_fit_local_phase': evaluate(decode_structured(other), other),
                       'shift_crc': shifted}
                for name, trace in (('h1_phase', original), ('local_phase', local)):
                    conditional = []
                    remaining = trace[48:].reshape(250, 3)
                    for order in shuffle:
                        moved = np.r_[trace[:48], remaining[order].ravel()]
                        conditional.append(evaluate(decode_structured(moved), moved)['crc_ok'])
                    row[name]['conditional_control_passes'] = np.flatnonzero(conditional).tolist()
                rows.append(row)
            print(group, [(r['tick'], r['h1_phase']['crc_ok'], r['local_phase']['crc_ok'])
                          for r in rows if r['group'] == group], flush=True)

    positives = []
    with np.load(artifact(comparison, 'E010', 'full_header_observations.npz'), allow_pickle=False) as source:
        for row in control['frames']:
            frame = row['capture_ordinal']
            soft = -source[f'frame_{frame:02d}_h2_soft'].astype(complex)
            trial = audit_header(soft, triplets)
            positives.append({'frame': frame, 'matches_archived_words': trial['words_hex'] == row['second_header']['words_hex'],
                              **trial})
    print('positive control matches', sum(r['matches_archived_words'] for r in positives), '/', len(positives), flush=True)
    result = {'schema': 'starlink-ut-offset-constrained-h2/v1', 'seed': SEED, 'permutation_orders': shuffle.tolist(),
              'triplet_folds_after_prefix': triplets[48::3].tolist(),
              'h2_start_resource': start, 'input_sha256': {p.as_posix(): h for p, h in inputs.items()},
              'weak_frames': rows, 'independent_capture_positive_control': positives}
    out = experiment_output('E109')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ut_offset_constrained_h2.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
