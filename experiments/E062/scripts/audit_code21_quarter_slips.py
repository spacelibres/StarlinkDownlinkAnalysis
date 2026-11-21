"""Test Code 21 words for OFDM-slot quadrant changes using held parity blocks."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code28_reencoding import fft_binary_parities
from audit_qin_exemplars import CALIBRATION_FRAME, remove_header_gain
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, equalize_grid
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from validate_code21 import LABELS, LIFT, K as SYSTEM_BITS, N, PHASE_ZERO, coordinates


WORD_SYMBOLS = N//2
CHECK_BITS = N-SYSTEM_BITS
BLOCKS = CHECK_BITS//LIFT
ROTATION = np.array([1., 1j, -1., -1j])
WEIGHTS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
GROUP_BITS = 32
SEED = 2026112162


def calibration_groups():
    rng = np.random.default_rng(SEED)
    groups = np.arange(CHECK_BITS//GROUP_BITS).reshape(-1, 3)
    assignment = np.array([rng.permutation(bundle) for bundle in groups])
    return assignment.T.tolist()


GROUPS = calibration_groups()


def apply_phase(waveform, slots, states):
    _, group = np.unique(slots, return_inverse=True)
    return waveform*ROTATION[np.asarray(states)[group]]


def syndrome_words(bits, generator, parity_shift=0):
    predicted, _ = fft_binary_parities(bits[:, :SYSTEM_BITS].reshape(-1, SYSTEM_BITS//LIFT, LIFT), generator)
    received = bits[:, SYSTEM_BITS:]
    if parity_shift:
        received = np.roll(received.reshape(-1, BLOCKS, LIFT), parity_shift, axis=2).reshape(-1, CHECK_BITS)
    return received ^ predicted


def phase_deltas(values, slots, orientation, generator, parity_shift=0):
    """Compute syndrome changes from one slot at a time, before selecting phases."""
    rotated = coordinates(values[None, :]*ROTATION[:, None], *orientation)
    relations = np.unique(slots)
    batch = np.broadcast_to(rotated[0], (1+3*len(relations), N)).copy()
    for i, slot in enumerate(relations):
        bit_positions = np.repeat(slots == slot, 2)
        batch[1+3*i:1+3*(i+1), bit_positions] = rotated[1:, bit_positions]
    synd = syndrome_words(batch, generator, parity_shift)
    deltas = np.zeros((len(relations), 4, CHECK_BITS), dtype=np.uint8)
    deltas[:, 1:] = (synd[1:] ^ synd[0]).reshape(-1, 3, CHECK_BITS)
    return synd[0], deltas


def phase_search(base, deltas, *, uniform):
    n_slots = len(deltas)
    if uniform:
        states = np.repeat(np.arange(4)[:, None], n_slots, axis=1)
    else:
        states = np.indices((4,)*n_slots).reshape(n_slots, -1).T
    current = np.broadcast_to(np.packbits(base), (len(states), CHECK_BITS//8)).copy()
    packed = np.packbits(deltas, axis=-1)
    for i in range(n_slots):
        current ^= packed[i, states[:, i]]
    errors = WEIGHTS[current].reshape(len(states), CHECK_BITS//GROUP_BITS, GROUP_BITS//8).sum(axis=2)
    rows = []
    for pair in GROUPS:
        training = errors[:, pair].sum(axis=1)
        minima = np.flatnonzero(training == training.min())
        chosen = minima[0]
        held = int(errors[chosen].sum()-training[chosen])
        rows.append({
            'calibration_groups': list(pair), 'candidate_count': len(states),
            'training_errors': int(training[chosen]), 'tied_minima': len(minima),
            'held_errors': held if len(minima) == 1 else None,
            'selected_states': states[chosen].tolist() if len(minima) == 1 else None,
        })
    accepted = rows[0]['selected_states']
    validated = accepted is not None and all(
        r['selected_states'] == accepted and r['training_errors'] == 0 and r['held_errors'] == 0 for r in rows
    )
    return {'validated': validated, 'accepted_states': accepted if validated else None,
            'baseline_errors': int(base.sum()), 'folds': rows}


def single_symbol_index(parity_generator):
    """Build a lookup of QPSK symbol changes for each training prefix."""
    check_columns = np.concatenate((parity_generator, np.eye(CHECK_BITS, dtype=np.uint8)))
    both = check_columns[::2] ^ check_columns[1::2]
    changes = np.concatenate((check_columns, both))
    positions = [(n,) for n in range(len(check_columns))]
    positions.extend((n, n+1) for n in range(0, len(check_columns), 2))
    tables = []
    for groups in GROUPS:
        visible = np.isin(np.arange(CHECK_BITS)//GROUP_BITS, groups)
        lookup = defaultdict(list)
        for index, row in enumerate(np.packbits(changes[:, visible], axis=1)):
            if row.any():
                lookup[row.tobytes()].append(index)
        tables.append((visible, lookup))
    return positions, np.packbits(changes, axis=1), tables


def try_single_symbol(observed, bank):
    """Select on training syndrome only and compare the remaining syndrome bits."""
    if not observed.any():
        return {'status': 'no_change', 'validated': False}
    positions, packed, tables = bank
    rows, compatible = [], set()
    for visible, lookup in tables:
        candidates = lookup.get(np.packbits(observed[visible]).tobytes(), [])
        selected = candidates[0] if len(candidates) == 1 else None
        row = {'training_candidates': len(candidates), 'selection': selected,
               'held_errors_after': None, 'held_affected_checks': None}
        if selected is not None:
            correction = np.unpackbits(packed[selected])
            remaining = np.count_nonzero(correction[~visible] ^ observed[~visible])
            row.update(held_errors_after=int(remaining), held_affected_checks=int(correction[~visible].sum()))
            if not remaining:
                compatible.add(selected)
        rows.append(row)
    first = rows[0]['selection']
    validated = first is not None and all(
        r['selection'] == first and r['held_errors_after'] == 0 and r['held_affected_checks'] > 0 for r in rows
    )
    return {
        'status': 'validated_held' if validated else 'no_held_validation', 'validated': validated,
        'folds': rows, 'selected_bit_positions': list(positions[first]) if validated else None,
        'same_word_compatible_local_hypothesis': list(positions[next(iter(compatible))])
        if len(compatible) == 1 and not validated else None,
    }


def phase_paths(soft, hard, sss, p, signs):
    slots = np.arange(2, len(soft)+1)
    pilot = equalize_grid(soft[1:, LOADED % 1024], soft[0, LOADED % 1024], sss, slots)
    return {
        'hard_direct': remove_header_gain(hard[1:len(soft), NONPILOT % 1024]*p[:len(slots)], signs),
        'soft_direct': remove_header_gain(soft[1:, NONPILOT % 1024]*p[:len(slots)], signs),
        'soft_pilot': remove_header_gain(pilot['nonpilot'], signs),
    }


def inspect_word(values, start, orientation, generator, bank):
    stop = start+WORD_SYMBOLS
    if not 1 <= start or stop >= len(values):
        raise ValueError('Expected complete Code 21 word and offset controls')
    slots = np.arange(start, stop)//len(NONPILOT)+2
    base, changes = phase_deltas(values[start:stop], slots, orientation, generator)
    uniform = phase_search(base, changes, uniform=True)
    varying = phase_search(base, changes, uniform=False)
    one_symbol = try_single_symbol(base, bank)
    controls = {}
    for delta in (-1, 1):
        position = start+delta
        indices = np.arange(position, position+WORD_SYMBOLS)
        args = phase_deltas(values[indices], indices//len(NONPILOT)+2, orientation, generator)
        controls[f'resource_shift_{delta:+d}'] = {**phase_search(*args, uniform=False),
                                                'one_symbol': try_single_symbol(args[0], bank)}
    args = phase_deltas(values[start:stop], slots, orientation, generator, parity_shift=1)
    controls['within_check_block_shift_+1'] = {**phase_search(*args, uniform=False),
                                             'one_symbol': try_single_symbol(args[0], bank)}
    result = {'slots': np.unique(slots).tolist(), 'baseline_errors': int(base.sum()),
              'uniform': uniform, 'per_slot': varying, 'one_symbol': one_symbol, 'controls': controls}
    words = {}
    if varying['validated']:
        selected = varying['accepted_states']
        corrected = apply_phase(values[start:stop], slots, selected)
        b = coordinates(corrected[None, :], *orientation)
        if syndrome_words(b, generator).any():
            raise ValueError('The accepted QPSK word fails full re-encoding')
        words['per_slot'] = b[0]
    if one_symbol['validated']:
        bits = coordinates(values[start:stop][None, :], *orientation)[0]
        flip = one_symbol['selected_bit_positions']
        symbol = flip[0]//2
        original = int(2*bits[2*symbol]+bits[2*symbol+1])
        bits[flip] ^= 1
        corrected = int(2*bits[2*symbol]+bits[2*symbol+1])
        if syndrome_words(bits[None, :], generator).any():
            raise ValueError('The accepted one-symbol hypothesis fails full re-encoding')
        point = values[start+symbol]
        quarter = np.arange(4)
        labels = LABELS[(orientation[0]*quarter+orientation[1]) % 4]
        lut = np.empty(4, dtype=complex)
        lut[labels] = np.exp(1j*(PHASE_ZERO+quarter*np.pi/2))
        slot, q = divmod(start+symbol, len(NONPILOT))
        one_symbol.update(
            resource=start+symbol, ofdm_slot=slot+2, carrier=int(NONPILOT[q]),
            original_label=original, corrected_label=corrected,
            observed_complex=[float(point.real), float(point.imag)],
            unit_radius_squared_distance_increase=float(abs(point-lut[corrected])**2-abs(point-lut[original])**2),
        )
        words['one_symbol'] = bits
    return result, words


def consistency(rows):
    phase = defaultdict(list)
    for row in rows:
        for branch, data in row['branches'].items():
            if data['per_slot']['validated']:
                for slot, state in zip(data['slots'], data['per_slot']['accepted_states'], strict=True):
                    phase[row['frame'], branch, slot].append(state)
    return [{'frame': frame, 'branch': branch, 'slot': slot, 'states': states, 'consistent': len(set(states)) == 1}
            for (frame, branch, slot), states in phase.items() if len(states) > 1]


def main():
    require_run_context()
    inputs = {}
    def checked(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    archive_base = checked('v032-qin-snapshot', 'E034', 'qin_soft_988.npz')
    hard_path = checked('v032-qin-snapshot', 'E034', 'qin_hard_988.npz')
    earlier = read(checked('v032-qin-exemplar-audit', 'E034', 'qin_exemplar_audit.json'))
    meta = {r['frame']: r for r in earlier['headers']}
    baseline = {r['frame']: r['branches'] for r in earlier['code21']}
    sss_path = ROOT/'references/qin/sssVec.mat'
    inputs[sss_path] = sha256(sss_path)
    if inputs[sss_path] != earlier['inputs_sha256'][sss_path.relative_to(ROOT).as_posix()]:
        raise ValueError('SSS reference changed')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    with np.load(checked('v014-code21-repeat', 'E016', 'code21_observations.npz'), allow_pickle=False) as ref:
        dense = ref['parity_generator']
        generator = dense[::LIFT].reshape(-1, BLOCKS, LIFT)
    bank = single_symbol_index(dense)
    signs = 1-2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    rows, samples = [], {}
    with np.load(archive_base, allow_pickle=False) as soft, np.load(hard_path, allow_pickle=False) as hard:
        if not np.array_equal(soft['frame'], hard['frame']):
            raise ValueError('Hard and soft archive frames are mispaired')
        p = np.column_stack([pn_reference(int(k), np.arange(2, soft['symbols'].shape[1]+1))
                             for k in NONPILOT]).conj()
        for f, y, z in zip(soft['frame'], soft['symbols'], hard['symbols'], strict=True):
            frame = int(f)
            if frame not in baseline:
                continue
            record = meta[frame]
            entry = record['h2']['layout20']['entries'][0]
            if entry['mcs_candidate'] != 21:
                raise ValueError('Expected Code 21 at the first archived allocation')
            start = record['h2_start']+114*record['h2']['blocks']
            paths = phase_paths(y, z, sss, p, signs)
            for word in range(entry['quantity']):
                row = {'frame': frame, 'word': word, 'start_resource': start+WORD_SYMBOLS*word,
                       'prior_orientation_calibration_frame': frame == CALIBRATION_FRAME, 'branches': {}}
                for key, values in paths.items():
                    old_key = 'pilot' if key == 'soft_pilot' else 'direct'
                    orientation = earlier['code21_orientation_models'][old_key]['orientation']
                    result, accepted = inspect_word(values, row['start_resource'], orientation, generator, bank)
                    if key != 'hard_direct' and result['baseline_errors'] != baseline[frame][old_key]['errors_by_word'][word]:
                        raise ValueError(f'Prior same-resource results differ in frame {frame}, word {word}')
                    row['branches'][key] = result
                    for method, bits in accepted.items():
                        field = f'frame_{frame}_word_{word}_{key}_{method}'
                        samples[field] = np.packbits(bits)
                        result[method]['validated_word_npz_key'] = field
                rows.append(row)
            print(frame, 'evaluated', entry['quantity'], flush=True)
    out = experiment_output('E062')
    out.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'starlink-code21-quarter-slips/v3',
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        'seed': SEED, 'training_groups_by_fold': GROUPS, 'group_width_bits': GROUP_BITS,
        'word_symbols': WORD_SYMBOLS, 'codeword_bit_pack_order': 'big',
        'h2_descriptor_cover': 'Qin archive frames 988 to 1000 with Code 21 as the first H2 allocation',
        'words': rows, 'adjacent_word_shared_slot_states': consistency(rows),
    }
    (out/'code21_quarter_slips.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out/'code21_quarter_validated_words.npz', **samples)


if __name__ == '__main__':
    main()
