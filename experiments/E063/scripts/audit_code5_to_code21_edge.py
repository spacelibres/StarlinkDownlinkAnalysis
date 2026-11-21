"""Test Code 5-derived edge phase and held-check erasures on Code 21."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code21_quarter_slips import (
    CHECK_BITS, GROUP_BITS, GROUPS, ROTATION, SYSTEM_BITS, WORD_SYMBOLS, phase_paths, syndrome_words,
)
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, recover_generator
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from sparse_code5_core import binary_rank, normalized_min_sum
from validate_code21 import LIFT, coordinates


CALIBRATION_CARRIERS = (-512, -511, -510, 509, 510, 511)
ERASURE_CARRIERS = (511, 510, -512)
H1_LENGTH = 228
CODE5_LENGTH = 2048
CODE5_SYSTEMATIC = 1024
MIN_PHASE_POINTS = 3
MIN_COHERENCE = .6


def learn_phase(signed):
    signed = np.asarray(signed, dtype=complex)
    if len(signed) < MIN_PHASE_POINTS or not abs(signed).sum():
        return None
    total = signed.sum()
    coherence = abs(total)/abs(signed).sum()
    if coherence < MIN_COHERENCE:
        return None
    scores = (ROTATION*total).real
    top = np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-9, rtol=1e-9))
    if len(top) != 1:
        return None
    return {'references': len(signed), 'coherence': float(coherence), 'quarter_turn': int(top[0]),
            'correction_angle': float(-np.angle(total)), 'response_gain': abs(total/len(signed))}


def recover_systematic_erasures(rows, syndrome, parity_observed, training_masks):
    trials = []
    guesses = []
    for mask in training_masks:
        training = parity_observed & mask
        held = parity_observed & ~mask
        fit = recover_generator(rows[:, training].T, syndrome[training, None])
        unique = fit['generator'] is not None
        trial = {
            'fit_rank': fit['rank'], 'fit_observed_equations': int(training.sum()),
            'fit_conflicts': fit['inconsistent_equations'], 'held_observed_equations': int(held.sum()),
            'held_rank': binary_rank(rows[:, held].T), 'held_disagreements': None,
        }
        if unique:
            guess = fit['generator'].ravel()
            trial['held_disagreements'] = int(np.count_nonzero(((guess@rows[:, held]) & 1) != syndrome[held]))
            guesses.append(guess)
        trials.append(trial)
    validated = bool(len(guesses) == len(training_masks) and all(
        row['held_rank'] == len(rows) and row['held_disagreements'] == 0 for row in trials
    ) and all(np.array_equal(guesses[0], other) for other in guesses[1:]))
    return {'validated': validated, 'folds': trials, 'unobserved_parity_count': int((~parity_observed).sum()),
            'systematic_guess': guesses[0].tolist() if validated else None}


def carrier_axis(start, length):
    return NONPILOT[(start+np.arange(length)) % len(NONPILOT)]


def code5_references(flat, stop, matrix, checks):
    words = flat[H1_LENGTH:stop].reshape(-1, CODE5_LENGTH)
    references = defaultdict(list)
    receipts = []
    for number, y in enumerate(words):
        k = carrier_axis(H1_LENGTH+number*CODE5_LENGTH, CODE5_LENGTH)
        erased = np.flatnonzero(np.isin(k, CALIBRATION_CARRIERS))
        observed = (y.real < 0).astype(np.uint8)
        input_word = observed.copy()
        input_word[erased] = 0
        estimate = recover_generator(checks[:, erased], ((checks@input_word) & 1)[:, None])
        if estimate['generator'] is not None:
            complete = input_word
            complete[erased] = estimate['generator'].ravel()
            method = 'exact_erasure'
        else:
            llr = y.real.copy()
            llr[erased] = 0.
            decoded = normalized_min_sum(llr, checks)
            complete = decoded['bits'] if decoded['converged'] else None
            method = 'min_sum' if complete is not None else 'unresolved'
        receipt = {'word': number, 'method': method, 'erased_count': len(erased),
                   'exact_rank': estimate['rank'], 'exact_conflicts': estimate['inconsistent_equations']}
        if complete is not None:
            if not np.array_equal((complete[:CODE5_SYSTEMATIC]@matrix) & 1, complete[CODE5_SYSTEMATIC:]):
                raise ValueError('Leading Code 5 word fails independent re-encoding')
            receipt['changed_outside_erasure'] = int(np.count_nonzero((complete != observed) & ~np.isin(
                np.arange(CODE5_LENGTH), erased
            )))
            compensated = y[erased]*(1-2*complete[erased].astype(float))
            for col, value in zip(k[erased], compensated, strict=True):
                references[int(col)].append((number, value, method))
        receipts.append(receipt)
    return references, receipts


def phase_profiles(reference):
    fits = {}
    for carrier in CALIBRATION_CARRIERS:
        records = reference.get(carrier, [])
        signed = np.array([r[1] for r in records], dtype=complex)
        fit = learn_phase(signed)
        exact_only = learn_phase([value for _, value, method in records if method == 'exact_erasure'])
        ids = np.array([r[0] for r in records], dtype=int)
        hold = []
        for word in np.unique(ids):
            use = ids == word
            model = learn_phase(signed[~use])
            if model is None:
                continue
            turn = ROTATION[model['quarter_turn']]
            hold.append({'word': int(word), 'count': int(use.sum()),
                         'raw_sign_errors': int(np.count_nonzero(signed[use].real < 0)),
                         'corrected_sign_errors': int(np.count_nonzero((signed[use]*turn).real < 0))})
        fits[str(carrier)] = {'combined': fit, 'exact_only': exact_only, 'code5_word_holdout': hold}
    return fits


def missing_bits_decode(bits, edges, dense, ring, masks):
    unknown = np.flatnonzero(np.repeat(edges, 2))
    sys = unknown[unknown < SYSTEM_BITS]
    erased_parity = unknown[unknown >= SYSTEM_BITS]-SYSTEM_BITS
    observed = np.ones(CHECK_BITS, dtype=bool)
    observed[erased_parity] = False
    baseline = bits.copy()
    baseline[sys] = 0
    residue = syndrome_words(baseline[None, :], ring)[0]
    result = recover_systematic_erasures(dense[sys], residue, observed, masks)
    result['sys_positions'] = sys.tolist()
    result['unobserved_parity_positions'] = erased_parity.tolist()
    if not result['validated']:
        return result, None
    baseline[sys] = result['systematic_guess']
    observed_errors = syndrome_words(baseline[None, :], ring)[0]
    if observed_errors[observed].any():
        raise ValueError('Measured parity must match before filling erased observations')
    result['changed_original_systematic_positions'] = sys[bits[sys] != baseline[sys]].tolist()
    result['original_erased_parity_disagreements'] = int(observed_errors[~observed].sum())
    baseline[SYSTEM_BITS:][~observed] ^= observed_errors[~observed]
    return result, baseline


def test_word(y, start, orientation, phases, donor_phase, dense, ring, masks):
    k = carrier_axis(start, len(y))
    original = coordinates(y[None, :], *orientation)[0]
    before = syndrome_words(original[None, :], ring)[0]
    phase_rows = {}
    edge = phases['511']['combined']
    near = phases['510']['combined']
    exact = phases['511']['exact_only']
    models = {}
    if edge is not None:
        models.update(quarter=ROTATION[edge['quarter_turn']], continuous=np.exp(1j*edge['correction_angle']),
                      wrong_half_turn=-ROTATION[edge['quarter_turn']])
    if near is not None:
        models['use_neighbor_phase'] = ROTATION[near['quarter_turn']]
    if exact is not None:
        models['quarter_from_exact_only'] = ROTATION[exact['quarter_turn']]
    if donor_phase is not None:
        models['wrong_frame_continuous'] = np.exp(1j*donor_phase['correction_angle'])
    phase_word = None
    for name, correction in models.items():
        rotated = y.copy()
        rotated[k == 511] *= correction
        bits = coordinates(rotated[None, :], *orientation)[0]
        error_count = int(syndrome_words(bits[None, :], ring).sum())
        phase_rows[name] = {
            'full_syndrome': error_count,
            'changed_positions': np.flatnonzero(original != bits).tolist()
        }
        if name == 'continuous' and error_count == 0:
            phase_word = bits
    repaired = {}
    erasures = {}
    for col in ERASURE_CARRIERS:
        result, recovered = missing_bits_decode(original, k == col, dense, ring, masks)
        erasures[str(col)] = result
        if recovered is not None:
            repaired[col] = recovered
    return {'baseline_syndrome': int(before.sum()), 'phase_transfers': phase_rows,
            'erasures': erasures}, repaired, phase_word


def main():
    require_run_context()
    inputs = {}
    def checked(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    prior = read(checked('v032-qin-exemplar-audit', 'E034', 'qin_exemplar_audit.json'))
    case = read(checked('v060-code21-one-symbol', 'E062', 'code21_quarter_slips.json'))
    soft_file = checked('v032-qin-snapshot', 'E034', 'qin_soft_988.npz')
    hard_file = checked('v032-qin-snapshot', 'E034', 'qin_hard_988.npz')
    with np.load(checked('v010-code5-parity-02', 'E012', 'code5_observations.npz'), allow_pickle=False) as f:
        matrix = f['parity_generator']
    with np.load(checked('v011-sparse-code5', 'E013', 'sparse_code5_arrays.npz'), allow_pickle=False) as f:
        checks = f['check_matrix']
    with np.load(checked('v014-code21-repeat', 'E016', 'code21_observations.npz'), allow_pickle=False) as f:
        dense = f['parity_generator']
    ring = dense[::LIFT].reshape(-1, CHECK_BITS//LIFT, LIFT)
    masks = [np.isin(np.arange(CHECK_BITS)//GROUP_BITS, g) for g in GROUPS]
    sss_path = ROOT/'references/qin/sssVec.mat'
    inputs[sss_path] = sha256(sss_path)
    if inputs[sss_path] != prior['inputs_sha256'][sss_path.relative_to(ROOT).as_posix()]:
        raise ValueError('SSS fingerprint changed')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    signs = 1-2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    headers = {r['frame']: r for r in prior['headers']}
    words = defaultdict(list)
    for row in case['words']:
        words[row['frame']].append(row)
    outputs, frame_diagnostics, bit_records = [], [], {}
    held = sorted(frame for frame, locations in words.items() if not locations[0]['prior_orientation_calibration_frame'])
    donors = dict(zip(held, held[1:]+held[:1], strict=True))
    calibration_cache = {}
    with np.load(soft_file, allow_pickle=False) as soft, np.load(hard_file, allow_pickle=False) as hard:
        if not np.array_equal(soft['frame'], hard['frame']):
            raise ValueError('Mismatched receiver paths')
        sf = soft['symbols']
        p = np.column_stack([pn_reference(int(k), np.arange(2, sf.shape[1]+1)) for k in NONPILOT]).conj()
        for i, f in enumerate(soft['frame']):
            frame = int(f)
            if frame not in words:
                continue
            stop5 = headers[frame]['h2_start']
            paths = phase_paths(sf[i], hard['symbols'][i], sss, p, signs)
            for branch in ('soft_direct', 'soft_pilot'):
                flat = paths[branch]
                refs, receipts = code5_references(flat, stop5, matrix, checks)
                profiles = phase_profiles(refs)
                calibration_cache[frame, branch] = flat, profiles
                frame_diagnostics.append({'frame': frame, 'branch': branch, 'code5_words': receipts, 'phase': profiles})
    for (frame, branch), (flat, profiles) in calibration_cache.items():
        donor = donors.get(frame)
        donor_phase = None if donor is None else calibration_cache[donor, branch][1]['511']['combined']
        model = 'direct' if branch == 'soft_direct' else 'pilot'
        orientation = prior['code21_orientation_models'][model]['orientation']
        for location in words[frame]:
            start = location['start_resource']
            r, repaired, phase_word = test_word(flat[start:start+WORD_SYMBOLS], start, orientation, profiles,
                                                donor_phase, dense, ring, masks)
            saved = location['branches'][branch]
            if saved['baseline_errors'] != r['baseline_syndrome']:
                raise ValueError('Archived Code 21 baseline is no longer reproducible')
            r.update(frame=frame, word=location['word'], branch=branch, start_resource=start,
                     wrong_frame_donor=donor,
                     prior_calibration_frame=location['prior_orientation_calibration_frame'],
                     e062_branch_single_symbol=saved['one_symbol']['validated'],
                     e062_any_branch_validated=any(b['per_slot']['validated'] or b['one_symbol']['validated']
                                                 for b in location['branches'].values()))
            tag = f'{frame}_{location["word"]}_{branch}'
            for k, corrected in repaired.items():
                key = f'word_{tag}_{k:+d}'
                bit_records[key] = np.packbits(corrected, bitorder='big')
                r['erasures'][str(k)]['npz_key'] = key
            if phase_word is not None:
                key = f'phase_{tag}'
                bit_records[key] = np.packbits(phase_word, bitorder='big')
                r['phase_transfers']['continuous']['npz_key'] = key
            outputs.append(r)
        print(frame, branch, 'calibrated Code 5 and tested', len(words[frame]), 'Code 21 words', flush=True)
    result = {
        'schema': 'starlink-code5-to-code21-edge/v2',
        'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        'code5_edge_carriers': CALIBRATION_CARRIERS, 'code21_erasure_carriers': ERASURE_CARRIERS,
        'group_width_bits': GROUP_BITS, 'check_groups': GROUPS, 'concentration_minimum': MIN_COHERENCE,
        'code5': frame_diagnostics, 'code21': outputs,
    }
    out = experiment_output('E063')
    out.mkdir(parents=True, exist_ok=True)
    (out/'code5_to_code21_edge.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out/'code21_edge_reconstructed_words.npz', **bit_records)


if __name__ == '__main__':
    main()
