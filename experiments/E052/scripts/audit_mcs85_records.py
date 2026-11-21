"""Read record spacing from byte decisions in high-order modulation allocations."""

from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS
from audit_ut_repeated_data import checked_artifact, equality_predictions, observations, pn_bits, read


ANCHOR_E038 = bytes.fromhex('BC 6B 50 ED 00 D6 A2 CC E0 CD FF 0B')
WIDTH = 20
ANCHOR_ERRORS = 16
BIT_AGREEMENT = .86
DETECTION_ERRORS = 4
MIN_LENGTH_EXAMPLES = 4
POSITION_CONTROLS = (-4, -1, 1, 4)
WEIGHT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def system_bytes(hard, cutoff, reference):
    return np.packbits(hard[:, :cutoff] ^ reference[:cutoff], bitorder='little', axis=1)


def learn_template(streams, calibration):
    anchor = np.frombuffer(ANCHOR_E038, dtype=np.uint8)
    anchor = np.packbits(np.unpackbits(anchor, bitorder='big'), bitorder='little')
    frequency, seed_sites = [], {}
    for frame in calibration:
        windows = sliding_window_view(streams[frame], WIDTH)
        chosen = np.flatnonzero(WEIGHT[windows[:, -len(anchor):] ^ anchor].sum(axis=1) <= ANCHOR_ERRORS)
        if not len(chosen):
            raise ValueError(f'Missing anchor calibration examples in frame {frame}')
        seed_sites[str(frame)] = chosen.tolist()
        bits = np.unpackbits(windows[chosen, :, None], axis=2, bitorder='little')
        frequency.append(bits.mean(axis=0))
    probabilities = np.mean(frequency, axis=0)
    stable = np.maximum(probabilities, 1-probabilities) >= BIT_AGREEMENT
    stable[4:8] = False
    even = (np.arange(WIDTH) % 2 == 0)[:, None]
    as_bytes = lambda bits: np.packbits(bits, axis=1, bitorder='little').ravel()
    value = as_bytes((probabilities > .5) & stable)
    return value, as_bytes(stable & even), as_bytes(stable & ~even), seed_sites


def detect_records(data, value, detection, verification, max_distance=DETECTION_ERRORS):
    windows = sliding_window_view(data, WIDTH)
    mismatch = windows ^ value
    distance = WEIGHT[mismatch & detection].sum(axis=1)
    sites = np.flatnonzero(distance <= max_distance)
    verified = WEIGHT[mismatch[sites] & verification].sum(axis=1)
    rows = []
    for site, errors in zip(sites, verified, strict=True):
        row = windows[site].tobytes()
        rows.append({'byte_offset': int(site), 'observed_hex': row.hex(), 'label_hex': row[4:6].hex(),
                     'candidate_counter_be': int.from_bytes(row[6:8], 'big'),
                     'candidate_counter_le': int.from_bytes(row[6:8], 'little'),
                     'detection_bit_disagreements': int(distance[site]),
                     'verification_bit_disagreements': int(errors)})
    return rows


def fit_distances(results, calibration):
    counts = defaultdict(Counter)
    for frame in calibration:
        rows = results[frame]
        for first, second in zip(rows, rows[1:]):
            counts[first['label_hex']][second['byte_offset']-first['byte_offset']] += 1
    mapping = {}
    for label in sorted(counts, key=lambda label: (-sum(counts[label].values()), label))[:3]:
        scores = counts[label]
        if sum(scores.values()) < MIN_LENGTH_EXAMPLES:
            continue
        distance = min(scores, key=lambda step: (-scores[step], step))
        mapping[label] = distance
    return mapping, counts


def predict_successor(start, distance, byte_count, detected):
    target = start + distance
    if target + WIDTH > byte_count:
        return 'outside'
    return 'matched' if target in detected else 'not_detected'


def check_frame(rows, byte_count, mapping):
    detected = {r['byte_offset']: r for r in rows}
    controls = [{'type': 'permutation', 'distances': dict(zip(mapping, values, strict=True))}
                for values in permutations(mapping.values())]
    controls.extend({'type': 'constant', 'distances': dict.fromkeys(mapping, size)} for size in sorted(set(mapping.values())))
    controls.extend({'type': f'offset_{shift:+d}', 'distances': {tag: size+shift for tag, size in mapping.items()}}
                    for shift in POSITION_CONTROLS)
    longest = max(max(c['distances'].values()) for c in controls)
    details, steps = [], []
    common = [r for r in rows if r['label_hex'] in mapping and r['byte_offset']+longest+WIDTH <= byte_count]
    for row in rows:
        label = row['label_hex']
        if label in mapping:
            pos = row['byte_offset']
            gap = mapping[label]
            advance = predict_successor(pos, gap, byte_count, detected)
            details.append({'reference': pos, 'predicted': pos+gap, 'label': label, 'status': advance})
    for control in controls:
        hits = sum(predict_successor(r['byte_offset'], control['distances'][r['label_hex']], byte_count, detected)
                   == 'matched' for r in common)
        control.update(compared=len(common), matches=hits)
    for a, b in zip(rows, rows[1:]):
        first, second = a['byte_offset'], b['byte_offset']
        established = a['label_hex'] == b['label_hex'] and mapping.get(a['label_hex']) == second-first
        if established:
            delta = {order: (b[f'candidate_counter_{order}']-a[f'candidate_counter_{order}']) % 65536 for order in ('be', 'le')}
            steps.append({'from': first, 'to': second, 'label': a['label_hex'], 'step_mod65536': delta})
    return details, controls, steps


def main():
    require_run_context()
    inputs = {}
    datasets, split, model = observations(inputs)
    bounds = read(checked_artifact('v049-blind-pn-corrected', 'E051', 'blind_pn_boundaries.json', inputs))
    reference = lambda phase: equality_predictions(np.arange(16384), pn_bits(), {**model, 'phase': model['phase']+phase})
    models = {'pn15': reference(0), 'phase_minus_one': reference(-1),
              'phase_plus_one': reference(1), 'raw': np.zeros(16384, dtype=np.uint8)}
    streams = {}
    for frame in bounds['frames']:
        number = frame['frame']
        from_qam = datasets[85].bits[datasets[85].frames == number]
        regions = [r for r in frame['measured_layout_comparison'] if r['mcs'] == 85]
        if len(regions) != 1 or len(from_qam) != regions[0]['word_count']:
            raise ValueError('MCS 85 requires exactly one consistent allocation per frame')
        streams[number] = system_bytes(from_qam, SYSTEMATIC_ENDS[85], models['pn15']).ravel()
    value, mask, verify, seed_sites = learn_template(streams, split['calibration'])
    results = {frame: detect_records(data, value, mask, verify) for frame, data in streams.items()}
    mapping, distance_counts = fit_distances(results, split['calibration'])
    by_frame = []
    for frame, rows in results.items():
        words = len(streams[frame]) // (SYSTEMATIC_ENDS[85]//8)
        checks, controls, counters = check_frame(rows, len(streams[frame]), mapping)
        by_frame.append({'frame': frame, 'system_bytes': len(streams[frame]), 'words': words,
                         'records': rows, 'length_predictions': checks, 'position_controls': controls,
                         'candidate_count_steps': counters})

    transfer = []
    for code, (bits, groups, _, _) in sorted(datasets.items()):
        for frame in np.unique(groups):
            local = bits[groups == frame]
            output = {}
            for name, pn in models.items():
                data = system_bytes(local, SYSTEMATIC_ENDS[code], pn).ravel()
                output[name] = detect_records(data, value, mask, verify)
            data = system_bytes(local, SYSTEMATIC_ENDS[code], models['pn15']).ravel()
            output['reverse_template'] = detect_records(data, value[::-1], mask[::-1], verify[::-1])
            transfer.append({'mcs': code, 'frame': int(frame), 'byte_count': len(data),
                             'detections': {k: v for k, v in output.items() if k != 'pn15' or code != 85}})

    report = {'schema': 'starlink-mcs85-record-spacing/v1', 'frames_split': split, 'inherited_split_seed': 2026112128,
              'input_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}, 'pn_model': model,
              'window_bytes': WIDTH, 'anchor_e038_msb_first_hex': ANCHOR_E038.hex(), 'anchor_seed_max_errors': ANCHOR_ERRORS,
              'anchor_calibration_sites': seed_sites, 'stable_bit_min_agreement': BIT_AGREEMENT,
              'template_hex': value.tobytes().hex(), 'detection_mask_hex': mask.tobytes().hex(),
              'verification_mask_hex': verify.tobytes().hex(), 'detection_max_errors': DETECTION_ERRORS,
              'fitted_distances_by_exact_label': mapping, 'calibration_detected_successor_gaps': distance_counts,
              'frame_results': by_frame, 'format_transfer_and_controls': transfer,
              'scope': 'Evidence for repeated systematic-region records. Raw fields retained without parity correction.'}
    output = experiment_output('E052')
    output.mkdir(parents=True, exist_ok=True)
    (output/'mcs85_records.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Detected', sum(len(rows) for rows in results.values()), 'records; mapping', mapping, flush=True)


if __name__ == '__main__':
    main()
