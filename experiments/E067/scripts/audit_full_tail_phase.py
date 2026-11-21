"""Measure terminal families before comparing head-derived resource endpoints."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from allocation_core import BIT_MODELS, UNITS
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, equalize_grid, fft_grid
from layout_core import NONPILOT, eligible_first_headers
from receiver_models import pn_reference
from snapshot_qin_exemplars import ArchiveReader, ETAG, PAGE, SIZE, URL, as_complex


PERIOD = 60
TOTAL = 301200
MIN_SKILL = .80
MARGIN = .10
SEED = 2026112167
SLOTS = np.arange(2, 302)
LEGACY_RAW = ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
LEGACY_SHA = 'a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf'
ADDITIONAL_BITS = {35: (16384, 3), **{k: (16384, 5) for k in (71, 75, 78, 81, 85, 88, 92, 93)}}


class CacheOnly(ArchiveReader):
    def load_page(self, start):
        end = min(start+PAGE, SIZE)-1
        if not (self.cache/f'{start}-{end}.bin').is_file():
            raise FileNotFoundError(f'Published archive range is not cached: {start}-{end}')
        return super().load_page(start)


def phase_bank(word, pos):
    return word[(pos[None, :]-np.arange(PERIOD)[:, None]) % PERIOD]


def train_and_hold(y, bank):
    training = y[-PERIOD:]
    held = y[-2*PERIOD:-PERIOD]
    gains = bank.conj() @ training / PERIOD
    independent = bank.conj() @ held / PERIOD
    best = int(abs(gains).argmax())
    held_best = int(abs(independent).argmax())
    prediction = gains[best]*bank[best]
    energy = max(np.vdot(held, held).real, 1e-30)
    reduction = float(1-np.vdot(held-prediction, held-prediction).real/energy)
    unique = np.count_nonzero(np.isclose(abs(gains), abs(gains[best]), rtol=1e-8, atol=1e-12)) == 1
    unique &= np.count_nonzero(np.isclose(abs(independent), abs(independent[held_best]),
                                         rtol=1e-8, atol=1e-12)) == 1
    errors = int(np.count_nonzero((held*gains[best].conj()).real*bank[best] < 0))
    return best, held_best, bool(unique), gains[best], reduction, errors


def coarse_onset(y, signs, gain):
    discrepant = (y*gain.conj()).real*signs <= 0
    tails = np.r_[0., np.cumsum(np.where(discrepant[::-1], np.log(.03/.5), np.log(.97/.5)))]
    count = int(tails.argmax())
    errors = int(discrepant[len(y)-count:].sum())
    coarse = len(y)-count if count >= 60 and 10*errors <= count else None
    return coarse, count, errors


def classify_tail(values, word):
    n = len(values)
    bank = phase_bank(word, np.arange(n-PERIOD, n))
    phase, held_phase, unique, gain, periodic, periodic_errors = train_and_hold(values, bank)
    _, _, _, constant_gain, const, constant_errors = train_and_hold(values, np.ones((1, PERIOD)))
    periodic_ok = periodic >= MIN_SKILL and periodic-const >= MARGIN and unique and phase == held_phase
    constant_ok = const >= MIN_SKILL and const-periodic >= MARGIN
    kind, coarse, refined = None, None, None
    count, errors = None, None
    if periodic_ok:
        kind = 'periodic'
        pos = np.arange(n-2*PERIOD)
        coarse, count, errors = coarse_onset(values[:len(pos)], word[(pos-phase) % PERIOD], gain)
        if coarse is not None:
            refined = coarse+(phase-coarse+PERIOD//2) % PERIOD-PERIOD//2
            if not 0 <= refined <= len(pos):
                refined = None
    elif constant_ok:
        kind = 'constant'
        earlier = values[:-2*PERIOD]
        coarse, count, errors = coarse_onset(earlier, np.ones(len(earlier)), constant_gain)
    return {'kind': kind, 'periodic_phase': phase, 'independently_selected_held_phase': held_phase,
            'both_phase_fits_unique': unique, 'periodic_held_skill': periodic, 'constant_held_skill': const,
            'periodic_held_sign_errors': periodic_errors, 'constant_held_sign_errors': constant_errors,
            'coarse_start': coarse, 'refined_start': refined, 'prepend_confirmed_resources': count,
            'prepend_sign_errors': errors}


def short_tail(values, words):
    pos = np.arange(len(values)-40, len(values))
    coords = (pos[None, :]-np.arange(PERIOD)[:, None]) % PERIOD
    waves = words[:, coords]
    x = values[-40:-20]
    y = values[-20:]
    a = waves[:, :, :20] @ x/20
    b = waves[:, :, 20:] @ y/20
    fit = abs(a)
    hold = abs(b)
    row = np.arange(len(words))
    phase = fit.argmax(axis=1)
    verify = hold.argmax(axis=1)
    unique = (np.isclose(fit, fit[row, phase, None], atol=1e-12, rtol=1e-8).sum(axis=1) == 1
              ) & (np.isclose(hold, hold[row, verify, None], atol=1e-12, rtol=1e-8).sum(axis=1) == 1)
    fit_coherence = fit[row, phase]/max(np.mean(abs(x)), 1e-30)
    held_coherence = hold[row, verify]/max(np.mean(abs(y)), 1e-30)
    error = np.count_nonzero((y*a[row, phase, None].conj()).real*waves[row, phase, 20:] <= 0, axis=1)
    passed = unique & (phase == verify) & (fit_coherence >= .90) & (held_coherence >= .90) & (error <= 1)
    return {'accepted': bool(passed[0]), 'phase': int(phase[0]), 'held_phase': int(verify[0]),
            'fit_coherence': float(fit_coherence[0]), 'held_coherence': float(held_coherence[0]),
            'held_sign_errors': int(error[0]),
            'accepted_shuffled_indices': np.flatnonzero(passed[1:]).tolist()}


def matched_tone_tail(values, word):
    values = np.asarray(values).reshape(4, -1)
    pos = np.arange(TOTAL-values.size, TOTAL)
    train = np.array([1, 3])
    held = np.array([0, 2])
    bank = phase_bank(word, pos).reshape(PERIOD, *values.shape)
    response = np.mean(bank[:, train]*values[train], axis=1)
    phase = int(np.argmax(np.sum(abs(response)**2, axis=1)))
    y = values[held]
    energy = max(np.vdot(y, y).real, 1e-30)
    periodic_error = y-response[phase]*bank[phase, held]
    constant_error = y-values[train].mean(axis=0)
    periodic = float(1-np.vdot(periodic_error, periodic_error).real/energy)
    const = float(1-np.vdot(constant_error, constant_error).real/energy)
    kind = None
    if const >= MIN_SKILL and const-periodic >= MARGIN:
        kind = 'constant'
    elif periodic >= MIN_SKILL and periodic-const >= MARGIN:
        kind = 'periodic'
    return {'kind': kind, 'periodic_phase': phase, 'periodic_held_skill': periodic, 'constant_held_skill': const}


def endpoint_models(start, blocks, descriptors, widths):
    separate = joint = start+114*blocks
    bits = {**BIT_MODELS, **ADDITIONAL_BITS}
    for entry in descriptors:
        code, n = entry['mcs_candidate'], entry['quantity']
        separate += n*widths[code]
        if code in bits:
            count, bps = bits[code]
            joint += (n*count+bps-1)//bps
        else:
            joint += n*widths[code]
    return separate, joint


def measured_frame(frame, group, values, wrong, word, controls, packed):
    normal = classify_tail(values, word)
    negative = classify_tail(wrong, word)
    tones = matched_tone_tail(values[-4*len(NONPILOT):], word)
    wrong_tones = matched_tone_tail(wrong[-4*len(NONPILOT):], word)
    row = {'frame': int(frame), 'group': int(group), 'signal': normal, 'wrong_pn': negative, 'tail_row': len(packed),
           'matched_tones': tones, 'matched_tones_wrong_pn': wrong_tones}
    if normal['kind'] is None:
        row.update(secondary_short=short_tail(values, controls), secondary_short_wrong_pn=short_tail(wrong, controls))
    return row, values[-2*PERIOD:].copy()


def compare_endpoint(row, start, blocks, entries, widths):
    separate, joint = endpoint_models(start, blocks, entries, widths)
    signal = row['signal']
    period = signal['kind'] == 'periodic'
    coarse, refined = signal['coarse_start'], signal['refined_start']
    short = row.get('secondary_short')
    tone = row['matched_tones']
    return {'header_end': start+114*blocks, 'descriptors': entries,
            'separate_rounding_end': separate, 'joint_rounding_end': joint,
            'separate_closes_phase': separate % PERIOD == signal['periodic_phase'] if period else None,
            'joint_closes_phase': joint % PERIOD == signal['periodic_phase'] if period else None,
            'coarse_minus_separate': coarse-separate if coarse is not None else None,
            'refined_minus_separate': refined-separate if refined is not None else None,
            'short_separate_closes_phase': separate % PERIOD == short['phase'] if short and short['accepted'] else None,
            'matched_tone_separate_closes_phase':
                separate % PERIOD == tone['periodic_phase'] if tone['kind'] == 'periodic' else None}


def collect_qin(word, controls, pn, shifted):
    reader = CacheOnly(ROOT/'.cache/qin-mat-pages', False)
    results, packed = [], []
    with reader, h5py.File(reader, 'r') as mat:
        source = mat['yDataDec']
        if source.shape != (1009, 301, 1024):
            raise ValueError('Qin archive dimension changed')
        for start in range(0, 1009, source.chunks[0]):
            stop = min(start+source.chunks[0], 1009)
            raw_frames = as_complex(source[start:stop, 1:, :])[..., NONPILOT % 1024]
            for n, grid in enumerate(raw_frames, start):
                row, tail = measured_frame(n, start, (grid*pn).ravel(), (grid*shifted).ravel(), word, controls, packed)
                results.append(row)
                packed.append(tail)
            if start % 130 == 0 or stop == 1009:
                print(f'Qin signal-only {stop}/1009', flush=True)
    return results, packed, {'url': URL, 'etag': ETAG, 'bytes': SIZE,
                             'pages': [reader.receipts[k] for k in sorted(reader.receipts)]}


def collect_legacy(word, controls, pn, shifted, sss):
    iq = np.load(LEGACY_RAW, allow_pickle=False, mmap_mode='r')
    results, packed = [], []
    for n in eligible_first_headers(len(iq)):
        try:
            grid = fft_grid(iq, n)
        except IndexError as problem:
            results.append({'frame': n, 'signal': None, 'receiver_error': str(problem)})
            continue
        equal = equalize_grid(grid[1:], grid[0], sss, SLOTS)
        values = equal['nonpilot']
        row, tail = measured_frame(n, 0, values.ravel(), (values*shifted/pn).ravel(), word, controls, packed)
        row['held_pilot_evm'] = equal['reserved_pilot_evm']
        results.append(row)
        packed.append(tail)
        print('Legacy signal-only', n, row['signal']['kind'], flush=True)
    return results, packed


def main():
    require_run_context()
    inputs = {}
    def archive(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))
    reference = archive('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    word = np.asarray(reference['cycle_folds'][0]['signed_template'])
    rng = np.random.default_rng(SEED)
    controls = np.stack([word, *[rng.permutation(word) for _ in range(199)]])
    p = np.column_stack([pn_reference(int(k), SLOTS) for k in NONPILOT]).conj()
    wrong = np.column_stack([pn_reference(int(k), SLOTS+1) for k in NONPILOT]).conj()
    sss_file = ROOT/'references/qin/sssVec.mat'
    inputs[sss_file] = sha256(sss_file)
    sss = loadmat(sss_file)['sssVecFull'].ravel()[LOADED % 1024]
    inputs[LEGACY_RAW] = sha256(LEGACY_RAW)
    if inputs[LEGACY_RAW] != LEGACY_SHA:
        raise ValueError('Legacy IQ source changed')
    qin, q_tails, pages = collect_qin(word, controls, p, wrong)
    legacy, l_tails = collect_legacy(word, controls, p, wrong, sss)

    # Header records are available only after both signal paths have finished.
    q_headers = archive('v034-qin-header-survey', 'E036', 'qin_header_survey.json')
    l_headers = archive('v008-complete-header-layout-02', 'E010', 'full_header_layout.json')
    widths_receipt = archive('v033-qin-lengths-final', 'E035', 'qin_lengths.json')
    width = {**UNITS, **{int(k): v for field in ('inherited_units', 'fitted_units')
                        for k, v in widths_receipt[field].items()}, 93: 3277}
    for m, h in zip(qin, q_headers['frames'], strict=True):
        if m['frame'] != h['frame'] or not (h['h1']['crc_ok'] and h['h2']['crc_ok']
                                            and h['h2']['layout20']['packing_valid']):
            raise ValueError('Mismatched or unverified header')
        m['comparison'] = compare_endpoint(m, h['h2_start'], h['h2']['blocks'], h['h2']['layout20']['entries'], width)
    for m, h in zip(legacy, l_headers['frames'], strict=True):
        h2 = h['second_header']
        if m['frame'] != h['legacy_frame_index'] or not (h2['crc_ok'] and h['first_header']['crc_ok']
                                                         and h2['layout20']['packing_valid']):
            raise ValueError('Mismatched or unverified legacy header')
        if m['signal'] is not None:
            m['comparison'] = compare_endpoint(m, h2['start_resource'], h2['blocks'], h2['layout20']['entries'], width)
    prior_frames = [r['frame'] for r in widths_receipt['frames']]
    result = {'schema': 'full-tail-phase-audit/v2',
              'reference_word': word.tolist(), 'held_minimum_skill': MIN_SKILL, 'held_family_margin': MARGIN,
              'secondary_permutation_seed': SEED, 'secondary_shuffles': controls[1:].tolist(),
              'source_qin': pages, 'resource_widths': width, 'e035_prior_frames': prior_frames,
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
              'qin': qin, 'legacy': legacy}
    out = experiment_output('E067')
    out.mkdir(parents=True, exist_ok=True)
    (out/'tail_phase_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savez_compressed(out/'tail_observations.npz', qin=np.asarray(q_tails), legacy=np.asarray(l_tails))


if __name__ == '__main__':
    main()
