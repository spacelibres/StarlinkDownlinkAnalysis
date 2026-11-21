"""Search observed narrowband data for Code 5 boundaries without record offsets."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft, signal
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_later_records import load_evidence
from audit_code5_narrow_record_search import opportunities
from audit_code5_narrow_waveform_clock import (FS, IQ, PREFIX, SLOT_NARROW, SLOT_WIDE, SSS, SUBCARRIER_HZ,
                                             USEFUL, WIDE_FS, narrow_waveform)
from audit_code5_record_extents import (COUNT_BIT, COUNT_WIDTH, FOOTER_HEADER_BITS, MARKER, MARKER_BITS,
                                       MARKER_ORIGIN, buffer_at)
from audit_code5_orbit_state import file_hash
from audit_h2_projection import visible_positions
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from code5_core import sinc_sample
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference


CODE5_ORIGIN = 228
WORD_BITS = 1024
LEGACY_TUNINGS_HZ = (-80_000_000, -63_000_000, 55_000_000)
LEGACY_SAMPLE_BITS = np.r_[112:120, 136:152]
MINIMUM_LATER_RECORD_START = 20 + 135 + 13
LEGACY_MAX_WORDS = 8
ROLL_OFF_HZ = 250_000
AXIS_COHERENCE_MIN = .65


def scan_packed(value, measured, byte_count, marker=MARKER):
    count_mask = ((1 << COUNT_WIDTH) - 1) << COUNT_BIT
    found = []
    for offset in range(byte_count - 19):
        bit = 8 * offset
        mask = measured >> bit
        if mask & count_mask != count_mask:
            continue
        data = value >> bit
        count = (data & count_mask) >> COUNT_BIT
        if not count:
            continue
        pos = MARKER_ORIGIN + 60 * count
        marker_and_footer_length = (0xfff | (0xff << 16)) << pos
        if mask & marker_and_footer_length != marker_and_footer_length or (data >> pos) & 0xfff != marker:
            continue
        octets = (data >> (pos + MARKER_BITS + 4)) & 0xff
        length = (pos + MARKER_BITS + FOOTER_HEADER_BITS + 7) // 8 + octets
        if offset + length <= byte_count:
            found.append((offset, length))
    return found


def scan_legacy(hard, pn, tag=(0x60, 0)):
    found = set()
    for word, bits in enumerate(hard):
        for offset in range(128 - 18):
            indices = 8 * offset + LEGACY_SAMPLE_BITS
            if (bits[indices] < 0).any():
                continue
            first = CODE5_ORIGIN + 2048 * word
            if (first + indices[0]) // len(NONPILOT) != (first + indices[-1]) // len(NONPILOT):
                continue
            header = np.packbits(bits[indices] ^ pn[indices], bitorder='little')
            for polarity in (0, 255):
                lead, a, b = map(int, header ^ polarity)
                count = lead >> 4
                length = 16 + 7 * count
                start = word * 128 + offset
                if (a, b) == tag and lead & 15 == 5 and count and start + length <= 128 * len(hard):
                    found.add((start, length))
    return found


def project_archive(words_by_frame, windows):
    maximum = max(max(words) for words in words_by_frame.values()) + 1
    prepared = {i: buffer_at(words, 0) for i, words in words_by_frame.items()}
    result = []
    for window in windows:
        geometry = [set(visible_positions(CODE5_ORIGIN + 2048 * k, WORD_BITS, window['carriers']))
                    for k in range(maximum)]
        candidates = []
        for frame, (stream, observed) in sorted(prepared.items()):
            keep = [seen and i % WORD_BITS in geometry[i // WORD_BITS] for i, seen in enumerate(observed)]
            mask = int.from_bytes(np.packbits(keep, bitorder='little').tobytes(), 'little')
            data = int.from_bytes(stream, 'little') & mask
            for label, word, marker in (('main', data, MARKER), ('wrong_marker', data, 0x200),
                                        ('reversed_polarity', data ^ mask, MARKER)):
                for start, length in scan_packed(word, mask, len(stream), marker=marker):
                    candidates.append({'frame': frame, 'start': start, 'length': length, 'scheme': label,
                                       'observed_counter_bits': ((mask >> (8 * start + 19)) & 0x3fff).bit_count(),
                                       'observed_type_bits': ((mask >> (8 * start)) & 0xff).bit_count(),
                                       'valid_preceding_layout': start >= MINIMUM_LATER_RECORD_START})
        result.append({'window': window['labels'], 'candidates': candidates})
    return result


def measure_record_geometry(windows, records):
    """Count direct carrier coverage for frozen record footprints, without filling erasures."""
    packed = []
    legacy = []
    for row in records['archive']['records']:
        parsed = row['parsed']
        if parsed is None or parsed['format'] != 'packed':
            continue
        marker_at = parsed['marker_bit']
        indices = np.r_[COUNT_BIT:COUNT_BIT+COUNT_WIDTH, marker_at:marker_at+MARKER_BITS,
                        marker_at+16:marker_at+24]
        packed.append(8 * row['start'] + indices)
    for row in records['480MSps']['records']:
        legacy.append(8 * row['start'] + LEGACY_SAMPLE_BITS)
    layouts = {}
    for name, positions in (('packed', packed), ('legacy', legacy)):
        coordinates = np.asarray(positions)
        resources = CODE5_ORIGIN + 2048 * (coordinates // WORD_BITS) + coordinates % WORD_BITS
        layouts[name] = NONPILOT[resources % len(NONPILOT)]

    output = []
    cache = {}
    for window in windows:
        carriers = tuple(sorted(window['carriers']))
        if carriers not in cache:
            cache[carriers] = {}
            for name, bins in layouts.items():
                counts = np.isin(bins, carriers).sum(axis=1)
                cache[carriers][name] = {
                    'records': len(bins), 'required_bits': bins.shape[1],
                    'complete': int(np.sum(counts == bins.shape[1])), 'max_covered_bits': int(counts.max()),
                }
        output.append({'kind': window['kind'], 'labels': window['labels'], 'carriers': carriers,
                       'record_geometry': cache[carriers]})
    return output


def estimate_legacy_bits(narrow, frame, scale, tune, sss):
    resources = CODE5_ORIGIN + 2048 * np.arange(LEGACY_MAX_WORDS)[:, None] + np.arange(WORD_BITS)
    slots = 2 + resources // len(NONPILOT)
    bins = NONPILOT[resources % len(NONPILOT)]
    used_slots = np.unique(slots)
    epoch = frame['epoch_25m']
    start = max(0, int(epoch) // 5 * 5 - 750)
    stop = min(len(narrow), int(epoch) + (int(slots.max()) + 2) * SLOT_NARROW + 750)
    region = narrow[start:stop]
    region = region[:len(region) // 5 * 5]
    interpolated = signal.resample(region, len(region) * 96 // 5)
    cfo = frame['cfo_hz']
    interpolated *= np.exp(2j * np.pi * (tune - cfo) * np.arange(len(interpolated)) / WIDE_FS)
    carriers = np.arange(-512, 512)
    in_passband = abs(carriers * SUBCARRIER_HZ - tune + cfo) < FS / 2 - ROLL_OFF_HZ
    carriers = carriers[in_passband]
    fits = [np.full(resources.shape, -1, dtype=np.int8) for _ in range(2)]

    def spectrum(slot):
        points = (epoch - start) * 96 / 5 + (slot * SLOT_WIDE + PREFIX + np.arange(USEFUL)) * scale
        if points[0] < 8 or points[-1] + 9 >= len(interpolated):
            return None
        return fft.fft(sinc_sample(interpolated, points))[carriers % USEFUL]

    sss_measurement = spectrum(1)
    if sss_measurement is None:
        return fits
    sync = sss[carriers % 1024]
    channel = np.divide(sss_measurement, sync, out=np.zeros_like(sss_measurement), where=abs(sync) > 0)
    present = abs(channel) > max(1e-12, .05 * np.median(abs(channel)))
    axes = [present & (carriers % 2 == which) for which in range(2)]
    reference = np.column_stack([pn_reference(int(k), used_slots) for k in carriers])
    for i, slot in enumerate(used_slots):
        sample = spectrum(slot)
        if sample is None:
            continue
        values = np.zeros(len(sample), dtype=np.complex128)
        values[present] = sample[present] * reference[i, present].conj() / channel[present]
        word, bit = np.nonzero((slots == slot) & np.isin(bins, carriers[present]))
        indices = np.searchsorted(carriers, bins[word, bit])
        for fit, axis in zip(fits, axes, strict=True):
            usable = axis & (abs(values) > 1e-12)
            if not usable.any():
                continue
            phasor = np.mean((values[usable] / abs(values[usable])) ** 2)
            if abs(phasor) < AXIS_COHERENCE_MIN:
                continue
            calibrated = values[indices] * np.exp(-.5j * np.angle(phasor))
            fit[word, bit] = calibrated.real < 0
    return fits


def search_legacy(raw, sss, pn):
    result = []
    for tune in LEGACY_TUNINGS_HZ:
        narrow = narrow_waveform(raw, tune, ROLL_OFF_HZ)
        try:
            frames, scale = opportunities(narrow, sss, tune)
        except ValueError as exc:
            if str(exc) != 'Insufficient PSS and SSS opportunities to fit sampling rate':
                raise
            result.append({'tune_hz': tune, 'status': str(exc), 'rows': []})
            continue
        rows = []
        for frame in frames:
            row = {**frame, 'candidates': [], 'wrong_tag_candidates': []}
            if frame['accepted']:
                halves = estimate_legacy_bits(narrow, frame, scale, tune, sss)
                row['candidates'] = sorted(set.intersection(*(scan_legacy(bits, pn) for bits in halves)))
                row['wrong_tag_candidates'] = sorted(set.intersection(*(scan_legacy(bits, pn, tag=(0x60, 1)) for bits in halves)))
            rows.append(row)
        result.append({'tune_hz': tune, 'scale': float(scale), 'rows': rows})
    return result


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    evidence = read(artifact('v098-code5-records-final', 'E100', 'code5_directory_records.json'))
    recovered, pages = load_evidence(artifact)
    if pages != evidence['archive_source']['pages']:
        raise ValueError('Qin cache page provenance changed')
    archive = project_archive(recovered['archive'], [w for w in evidence['leo_windows'] if w['kind'] == 'catalogued_nominal'])
    source = PROJECT_ROOT / IQ
    sync = PROJECT_ROOT / SSS
    inputs[source], inputs[sync] = file_hash(source), file_hash(sync)
    receiver = read(artifact('v010-code5-parity-02', 'E012', 'code5_parity.json'))
    if inputs[source] != receiver['input_sha256'] or inputs[sync] != receiver['sss_reference_sha256']:
        raise ValueError('Raw source or SSS reference differs from the frozen receiver evidence')
    model = read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(WORD_BITS), pn_bits(), model)
    raw = search_legacy(np.load(source, mmap_mode='r', allow_pickle=False), loadmat(sync)['sssVecFull'].ravel(), pn)

    # Ground truth is read only after both searches produce their candidates.
    truth = read(artifact('v139-code5-record-extents-final', 'E141', 'code5_record_extents.json'))['records']
    leo = read(artifact('v128-leo-full-code-observability', 'E130', 'leo_full_code_observability.json'))
    geometry = measure_record_geometry(leo['windows'], truth)
    qin_records = {(r['frame'], r['start'], r['length']) for r in truth['archive']['records']
                   if r['parsed'] is not None and r['parsed']['format'] == 'packed'}
    if any(r[1] < MINIMUM_LATER_RECORD_START for r in qin_records):
        raise ValueError('Frozen lower boundary excludes a valid later record')
    for w in archive:
        for row in w['candidates']:
            row['matches_archive'] = (row['frame'], row['start'], row['length']) in qin_records
        print(w['window'][0], Counter((r['scheme'], r['matches_archive'], r['valid_preceding_layout']) for r in w['candidates']), flush=True)
    epochs = {r['legacy_frame_index']: r['pss_start_sample'] * FS / WIDE_FS for r in
              read(artifact('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']}
    by_frame = {(r['frame'], r['start'], r['length']) for r in truth['480MSps']['records']}
    for w in raw:
        hits = 0
        wrong = 0
        for row in w['rows']:
            epoch = row['epoch_25m']
            frame = min(epochs, key=lambda k: abs(epochs[k] - epoch))
            if abs(epochs[frame] - epoch) > 40:
                frame = None
            row['wideband_frame'] = frame
            row['correct_candidates'] = [list(c) for c in row['candidates'] if (frame, *c) in by_frame]
            hits += len(row['correct_candidates'])
            wrong += len(row['candidates']) - len(row['correct_candidates'])
        print('Raw tune', w['tune_hz'], 'accepted', sum(r['accepted'] for r in w['rows']), 'right', hits, 'wrong', wrong, flush=True)
    result = {'schema': 'starlink-code5-blind-extents/v1', 'inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              'archive_page_receipts': pages, 'minimum_later_record_start': MINIMUM_LATER_RECORD_START,
              'leo_record_geometry': geometry,
              'qin_start_scan': archive, 'raw_480MSps_to_25MSps': raw}
    out = experiment_output('E142')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'code5_blind_extents.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
