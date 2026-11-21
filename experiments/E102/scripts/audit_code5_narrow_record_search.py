"""Acquire later Code 5 counters from a software-limited side band."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft, signal
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_code5_later_records import counter_from_record
from audit_code5_narrow_waveform_clock import (COARSE_TRIM, FS, IQ, PERIOD, PREFIX, SLOT_NARROW, SLOT_WIDE,
                                              SSS, SUBCARRIER_HZ, USEFUL, WIDE_FS, acquire, narrow_waveform,
                                              sync_templates)
from audit_code5_orbit_state import file_hash
from audit_leo_recording_coverage import COHORT, NATIVE_GLRT, carrier_if_hz
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from code5_core import sinc_sample
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import read_ci16


UT_IQ = 'data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin'
MAX_WORDS = 8
GROUP_SEED = 2026112202
MAX_COUNTER = 11250
RECORD_BITS = 40
REFERENCE_MIN_SCORE = 1.0
WITNESS_EPOCH_TOLERANCE = 40


def scan_word(equalized, pn, word):
    windows = np.lib.stride_tricks.sliding_window_view(equalized, RECORD_BITS)[::8]
    offsets = np.flatnonzero(np.isfinite(windows).all(axis=1))
    pn_windows = np.lib.stride_tricks.sliding_window_view(pn, RECORD_BITS)[::8]
    binary = (windows[offsets].real < 0).astype(np.uint8) ^ pn_windows[offsets]
    candidates = []
    for offset, header in zip(offsets, np.packbits(binary, axis=1, bitorder='little'), strict=True):
        if header[0] != 1:
            continue
        counter = counter_from_record(header.tobytes())
        if counter < MAX_COUNTER:
            candidates.append((word * 128 + int(offset), counter))
    return candidates


def select_origin(frames, direction):
    support = defaultdict(set)
    for candidates, ordinal, group in frames:
        for _, counter in candidates:
            support[counter - direction * ordinal].add(group)
    ranked = sorted(((len(groups), origin) for origin, groups in support.items()), reverse=True)
    unique = ranked and ranked[0][0] >= 2 and (len(ranked) < 2 or ranked[0][0] > ranked[1][0])
    return {'origin': ranked[0][1] if unique else None, 'ranked_group_support': ranked[:8]}


def narrow_ut(values, tune):
    rate = 250_000_000
    size = len(values) // 10 * 10
    time = np.arange(size, dtype=np.float64) / rate
    shifted = values[:size] * np.exp(-2j * np.pi * tune * time)
    return signal.resample(shifted, size // 10)


def opportunities(values, sss, tune):
    templates = sync_templates(sss, tune, 0)
    first = acquire(values, 0, int(PERIOD * 1.53), *templates)
    phase = (first['epoch_25m'] + PERIOD / 2) % PERIOD - PERIOD / 2
    if phase < -COARSE_TRIM:
        phase += PERIOD
    nominal = np.arange(phase, len(values) - 3 * SLOT_NARROW, PERIOD)
    rows = [acquire(values, max(0, int(e) - 80), int(e) + 330, *templates) for e in nominal]
    for i, row in enumerate(rows):
        row['ordinal'] = i
        row['accepted'] = row['score'] >= REFERENCE_MIN_SCORE
    held = [row for row in rows if row['accepted']]
    if len(held) < 3:
        raise ValueError('Insufficient PSS and SSS opportunities to fit sampling rate')
    slope, _ = np.polyfit([r['ordinal'] for r in held], [r['epoch_25m'] for r in held], 1)
    return rows, slope / PERIOD


def body_symbols(narrow, row, scale, tune, sss, pn, carriers, pilots):
    resources = 228 + 2048 * np.arange(1, MAX_WORDS)[:, None] + np.arange(1024)
    slots = 2 + resources // len(NONPILOT)
    bins = NONPILOT[resources % len(NONPILOT)]
    epoch = row['epoch_25m']
    start = max(0, int(epoch) // 5 * 5 - 750)
    stop = min(len(narrow), int(epoch) + (int(slots.max()) + 2) * SLOT_NARROW + 750)
    region = narrow[start:stop]
    region = region[:len(region) // 5 * 5]
    interp = signal.resample(region, len(region) * 96 // 5)
    interp *= np.exp(2j * np.pi * (tune - row['cfo_hz']) * np.arange(len(interp)) / WIDE_FS)
    body = np.full(resources.shape, np.nan + 1j * np.nan)

    def spectrum(slot):
        samples = (epoch - start) * 96 / 5 + (slot * SLOT_WIDE + PREFIX + np.arange(USEFUL)) * scale
        if samples[0] < 8 or samples[-1] + 9 >= len(interp):
            return None
        return fft.fft(sinc_sample(interp, samples))[carriers % USEFUL]

    measured_sss = spectrum(1)
    if measured_sss is None:
        return [], [], 0, None
    channel = measured_sss / sss[carriers % 1024]
    physical = abs(carriers * SUBCARRIER_HZ - tune + row['cfo_hz']) <= FS / 2
    visible = physical & (abs(channel) > .05 * np.median(abs(channel[pilots])))
    measured_slots = np.unique(slots)
    reference = np.column_stack([pn_reference(int(k), measured_slots) for k in carriers])
    evm = []
    for symbol, slot in enumerate(measured_slots):
        received = spectrum(slot)
        if received is None:
            continue
        calibrated = np.zeros(len(carriers), dtype=complex)
        calibrated[visible] = received[visible] * reference[symbol, visible].conj() / channel[visible]
        gain = np.mean(calibrated[pilots[::2]])
        if abs(gain) < 1e-12:
            continue
        calibrated /= gain
        evm.append(float(np.sqrt(np.mean(abs(calibrated[pilots[1::2]] - 1)**2))))
        k = bins - carriers[0]
        valid = (slots == slot) & (k >= 0) & (k < len(carriers))
        idx = np.nonzero(valid)
        observed = visible[k[idx]]
        subset = tuple(axis[observed] for axis in idx)
        body[subset] = calibrated[k[subset]]
    decoded, inverted = [], []
    for index, word in enumerate(body, 1):
        decoded.extend(scan_word(word, pn, index))
        inverted.extend(scan_word(-word, pn, index))
    return decoded, inverted, int(np.isfinite(body).sum()), float(np.median(evm)) if evm else None


def receiver(narrow, sss, tune, pn):
    rows, scale = opportunities(narrow, sss, tune)
    carriers = np.arange(398, 512) if tune > 0 else np.arange(-512, -397)
    pilots = (np.arange(488, 496) if tune > 0 else np.arange(-496, -488)) - carriers[0]
    accepted = [r for r in rows if r['accepted']]
    groups = np.arange(len(accepted)) // 2
    shuffled = np.random.default_rng(GROUP_SEED).permutation(np.unique(groups))
    selected = set(shuffled[:(len(shuffled) + 1) // 2])
    for row, group in zip(accepted, groups, strict=True):
        decoded, inverted, observed, evm = body_symbols(narrow, row, scale, tune, sss, pn, carriers, pilots)
        row.update(candidates=decoded, inverted_polarity_candidates=inverted, group=int(group),
                   calibration=bool(group in selected), available_system_bits=observed, pilot_evm=evm)
    calibration = [r for r in accepted if r['calibration']]
    origin = select_origin([(r['candidates'], r['ordinal'], r['group']) for r in calibration], 1)
    reverse = select_origin([(r['candidates'], r['ordinal'], r['group']) for r in calibration], -1)
    invert = select_origin([(r['inverted_polarity_candidates'], r['ordinal'], r['group']) for r in calibration], 1)
    for row in accepted:
        for label, model, direction, candidate_key in (
            ('forward', origin, 1, 'candidates'), ('reverse', reverse, -1, 'candidates'),
            ('inverted', invert, 1, 'inverted_polarity_candidates'),
        ):
            predicted = None if model['origin'] is None else model['origin'] + direction * row['ordinal']
            row[label] = {'predicted_counter': predicted, 'matched_offsets': [offset for offset, counter
                          in row[candidate_key] if counter == predicted]}
    return {'center_relative_to_bin_zero_hz': tune, 'estimated_sampling_ppm': float((scale - 1) * 1e6),
            'rows': rows, 'calibration_groups': sorted(map(int, selected)),
            'models': {'forward': origin, 'reverse': reverse, 'inverted': invert}}


def compare(arm, labels, epochs):
    frames = sorted(epochs)
    offsets = np.array([epochs[i] for i in frames]) * FS / WIDE_FS
    for row in arm['rows']:
        nearest = np.argmin(abs(offsets - row['epoch_25m']))
        frame = frames[nearest] if abs(offsets[nearest] - row['epoch_25m']) <= WITNESS_EPOCH_TOLERANCE else None
        row['wideband_frame'] = frame
        if not row['accepted']:
            continue
        witness = labels.get(frame)
        row['wideband_type_01'] = witness
        for label in ('forward', 'reverse', 'inverted'):
            model = row[label]
            model['matches_wideband'] = (witness is not None and model['predicted_counter'] == witness['record_counter']
                                         and witness['start'] in model['matched_offsets'])


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    leo = read(art('v069-leo-recording-coverage', 'E071', 'leo_recording_coverage.json'))
    leo_hashes = {path.as_posix(): file_hash(EXTERNAL_LEO_ROOT / path) for path in (COHORT, NATIVE_GLRT)}
    if leo_hashes != leo['sources_sha256']:
        raise ValueError('LEO tuning sources differ from E071')
    upper = leo['separate_native_glrt_capture']
    lower = leo['cohort'][0]
    tunes = {name: row['tuned_if_hz'] - carrier_if_hz(row['channel'], 0) for name, row in
             (('upper', upper), ('lower', lower))}
    paths = {key: ROOT / name for key, name in (('480MSps', IQ), ('250MSps', UT_IQ), ('sss', SSS))}
    for path in paths.values():
        inputs[path] = file_hash(path)
    legacy = read(art('v010-code5-parity-02', 'E012', 'code5_parity.json'))
    ut = read(art('v022-ut-replay', 'E014', 'ut_transfer.json'))
    if inputs[paths['480MSps']] != legacy['input_sha256'] or inputs[paths['250MSps']] != ut['source_sha256']:
        raise ValueError('Raw wideband recording SHA-256 mismatch')
    if inputs[paths['sss']] != legacy['sss_reference_sha256']:
        raise ValueError('SSS reference changed')
    sss = loadmat(paths['sss'])['sssVecFull'].ravel()
    model = read(art('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    results = {}
    for dataset, read_raw, render in (
        ('480MSps', lambda p: np.load(p, mmap_mode='r', allow_pickle=False),
         lambda values, tune: narrow_waveform(values, tune, 0)),
        ('250MSps', read_ci16, narrow_ut),
    ):
        arms = {}
        raw = read_raw(paths[dataset])
        for name, tune in tunes.items():
            narrow = render(raw, tune)
            arm = receiver(narrow, sss, tune, pn)
            arms[name] = arm
            print(dataset, name, 'PSS/SSS accepted', sum(r['accepted'] for r in arm['rows']),
                  'origin models', arm['models'], flush=True)
        results[dataset] = arms

    witnesses = read(art('v099-code5-later-records', 'E101', 'code5_later_records.json'))['records']
    old_frames = read(art('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']
    epochs = {'480MSps': {r['legacy_frame_index']: r['pss_start_sample'] for r in old_frames},
              '250MSps': {r['candidate_index']: r['pss']['epoch'] for r in ut['candidates']}}
    for dataset, arms in results.items():
        labels = {r['frame']: r for r in witnesses[dataset] if r['class'] == 'type_01'}
        for name, arm in arms.items():
            compare(arm, labels, epochs[dataset])
            for label in ('forward', 'reverse', 'inverted'):
                held = [r for r in arm['rows'] if r['accepted'] and not r['calibration']]
                print(dataset, name, label, 'held matches', sum(r[label]['matches_wideband'] for r in held),
                      'of', len(held), flush=True)
    record = {'schema': 'starlink-code5-narrow-record-search/v1',
              'inputs_sha256': {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              'external_leo_sha256': leo_hashes, 'max_words_from_resource_228': MAX_WORDS,
              'sync_score_minimum': REFERENCE_MIN_SCORE, 'group_seed': GROUP_SEED,
              'sample_rate_hz': FS, 'results': results}
    output = experiment_output('E102')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'code5_narrow_record_search.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
