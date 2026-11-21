"""Apply native LEO frequency offsets to the Code 5 clock observability model."""

from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_clock_correction import ANCHOR, CLOCK_BITS, fields, SCALE_DENOMINATOR, timestamp_numerator
from audit_code5_narrow_waveform_clock import (
    FS, IQ, SSS, SUBCARRIER_HZ, TARGET_CARRIERS, WORD_BITS, narrow_waveform, waveform_arm,
)
from audit_leo_pilot_archive import check
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


RECEIPT = Path('reports/figures/2026_09_03_0181_native25_fractional_glrt/analysis-summary.json')
RECEIPT_SHA256 = '5cddebaf8a77f2b4f9ca990e44c00293a79668c438e1d69ae1ae853d964b1cc9'
FREQUENCIES = ('acquired_cfo_hz', 'tracking_cfo_hz')


def missing_clock_bits(shift, nominal_511):
    baseband = nominal_511 + shift + (TARGET_CARRIERS - 511) * SUBCARRIER_HZ
    return tuple(bit for bit, center in zip(CLOCK_BITS, baseband, strict=True) if abs(center) >= FS / 2)


def complete_anchor(packed, missing):
    """List all clock tuples compatible with visible bits and E097's 15-second rule."""
    if len(missing) > 4 or not all(ANCHOR[0] <= bit < sum(ANCHOR) for bit in missing):
        return ()
    visible = packed & ~sum(1 << bit for bit in missing)
    found = []
    for assignment in product((0, 1), repeat=len(missing)):
        candidate = visible | sum(value << bit for bit, value in zip(missing, assignment, strict=True))
        clock = fields(candidate)
        if clock is not None:
            found.append(clock)
    return tuple(found)


def receipt_geometry(receipt, edge):
    if receipt['capture_id'] != edge['capture'] or receipt['method']['native_sample_rate_hz'] != FS:
        raise ValueError('Wrong native capture or sample rate')
    center = receipt['runtime']['frequency_reference_center_hz']
    native_rows = receipt['rows']
    if len(native_rows) != receipt['runtime']['trajectory_retained_count']:
        raise ValueError('Native row count differs from the receipt')
    nominal = edge['target_baseband_hz']
    first_epoch = native_rows[0]['fractional_global_epoch_sample']
    rows = []
    for row in native_rows:
        epoch = row['fractional_global_epoch_sample']
        window = row['window_start_sample']
        if row['fractional_epoch_status'] != 'complete' or not window <= epoch < window + FS / 50:
            raise ValueError('Retained epoch is outside the complete 20 ms window')
        cycles = (epoch - first_epoch) * 750 / FS
        frequencies = {}
        for method in FREQUENCIES:
            shift = row[method] - center
            frequencies[method] = {'shift_hz': shift, 'missing_clock_bits': missing_clock_bits(shift, nominal)}
        rows.append({'mode': row['mode_id'], 'window_start_sample': window, 'epoch_sample': epoch,
                     'relative_receive_ordinal': round(cycles), 'frequencies': frequencies})
    if not all(a['relative_receive_ordinal'] < b['relative_receive_ordinal'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Native epochs are not in increasing relative order')
    return {'nominal_511_baseband_hz': nominal, 'pilot_center_hz': center,
            'window_seconds': receipt['method']['native_window_ms'] / 1000, 'rows': rows}


def verdict(hard, target_ns, missing):
    packed = sum(int(bit) << index for bit, index in zip(hard, CLOCK_BITS, strict=True))
    candidates = complete_anchor(packed, missing)
    error = None
    if len(candidates) == 1:
        error = timestamp_numerator(*candidates[0]) - target_ns * SCALE_DENOMINATOR
    return {'missing_clock_bits': missing, 'valid_candidates': len(candidates), 'error_27ths_ns': error}


def measure_waveform(art, native):
    raw = PROJECT_ROOT / IQ
    reference = PROJECT_ROOT / SSS
    provenance = read(art('v010-code5-parity-02', 'E012', 'code5_parity.json'))
    raw_hash = check(raw, provenance['input_sha256'])
    reference_hash = check(reference, provenance['sss_reference_sha256'])
    model = read(art('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    prior = read(art('v096-code5-clock-cadence', 'E098', 'code5_narrow_waveform_clock.json'))
    baseline = prior['decoder_arms']['0/0']['frames']
    wide_cfo = int(np.median([row['cfo_hz'] for row in baseline]))
    nominal = native['nominal_511_baseband_hz']
    nominal_tune = 511 * SUBCARRIER_HZ - nominal
    shifts = [row['frequencies'][FREQUENCIES[0]]['shift_hz'] for row in native['rows']]
    low, center, high = (round(value) for value in np.quantile(shifts, [0, .5, 1]))
    iq = np.load(raw, mmap_mode='r', allow_pickle=False)
    sss = loadmat(reference)['sssVecFull'].ravel()
    arms = []
    for label, target, rolloff in (('native_min', low, 0), ('native_median', center, 0), ('native_max', high, 0),
                                    ('median_rolloff_250k', center, 250_000), ('median_rolloff_1m', center, 1_000_000)):
        render_shift = target - wide_cfo
        tune = nominal_tune - render_shift
        values = narrow_waveform(iq, tune, rolloff)
        arm = waveform_arm(values, sss, tune, rolloff, pn)
        if len(arm['frames']) != len(baseline) or any(abs(a['epoch_25m'] - b['epoch_25m']) > 4
                                                    for a, b in zip(arm['frames'], baseline)):
            raise ValueError('Filtered receiver selected a different 480 MS/s frame grid')
        arms.append({'label': label, 'rolloff_hz': rolloff, 'target_native_shift_hz': target,
                     'rendered_shift_hz': render_shift, 'frames': arm['frames']})

    expected = read(art('v093-code5-gps-epoch', 'E095', 'code5_gps_epoch.json'))['raw']['480MSps']['rows']
    for arm in arms:
        frames = []
        for sample, label in zip(arm.pop('frames'), expected, strict=True):
            hard = np.asarray(sample['decode']['hard_72'], dtype=np.uint8) ^ pn[WORD_BITS]
            shift = sample['cfo_hz'] + arm['rendered_shift_hz']
            target_ns = label['gps_ns']
            frames.append({'frame': label['frame'], 'epoch_25m': sample['epoch_25m'], 'shift_estimate_hz': shift,
                           'holdout_pilot_evm': sample['reserved_pilot_evm'], 'hard_clock_bits': hard.tolist(),
                           'direct': verdict(hard, target_ns, ()),
                           'top_two_erased': verdict(hard, target_ns, (518, 519)),
                           'top_four_erased': verdict(hard, target_ns, (516, 517, 518, 519)),
                           'reversed_polarity': verdict(hard ^ 1, target_ns, (518, 519))})
        arm['frames'] = frames
    return {'raw_sha256': raw_hash, 'sss_reference_sha256': reference_hash,
            'wideband_baseline_coarse_cfo_hz': wide_cfo, 'arms': arms}


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    source = EXTERNAL_LEO_ROOT / RECEIPT
    receipt_hash = check(source, RECEIPT_SHA256)
    edge = read(art('v069-leo-recording-coverage', 'E071', 'leo_recording_coverage.json'))['separate_native_glrt_capture']
    native = receipt_geometry(read(source), edge)
    wave = measure_waveform(art, native)
    values = {'schema': 'starlink-leo-native-clock-coverage/v1', 'receipt_sha256': receipt_hash,
              'prior_inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'capture': edge['capture'], 'grid': native, 'waveform_replay': wave}
    for arm in wave['arms']:
        counts = {}
        for method in ('direct', 'top_two_erased', 'top_four_erased', 'reversed_polarity'):
            errors = [row[method]['error_27ths_ns'] for row in arm['frames']]
            counts[method] = sum(error is not None and abs(error) <= 27 for error in errors)
        print(arm['label'], counts, flush=True)
    out = experiment_output('E107')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'leo_native_clock_coverage.json').write_text(json.dumps(values, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
