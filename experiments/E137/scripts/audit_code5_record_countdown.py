"""Recover the counted field after the optional Code 5 record list."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_byte_records import optional_structure, structure
from audit_code5_directory_checks import crc_bytes, raw_record_set
from audit_code5_frame_count import RELATIVE_BIT, counter_at
from audit_code5_record_trajectories import field_information, interval_bits
from audit_h2_projection import visible_positions
from audit_leo_early_symbol_roles import CODE5_LENGTH, CODE5_START
from audit_leo_layout_relation import archive_words
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from receiver_models import pn_bits


TAIL_BYTES = 69
TIMING_BYTE = 34
SYSTEM_BITS = 1024


def decode_timing_word(value):
    return {'countdown': (value >> 1) & 0x3f, 'groups': (value >> 9) & 7, 'other_bits': value & ~0x0e7e}


def countdown(frame_counter, period, phase, reload_phase):
    if reload_phase is not None and frame_counter % period == reload_phase:
        return period
    return (phase - frame_counter - 1) % period + 1


def tail_resource_positions(length, span, start_bit, width):
    first = 8 * (length + span - TAIL_BYTES + TIMING_BYTE) + start_bit
    return tuple(CODE5_START + (bit // SYSTEM_BITS) * CODE5_LENGTH + bit % SYSTEM_BITS
                 for bit in range(first, first + width))


def read_records(words, receipts, directory):
    if sorted(words) != [row['frame'] for row in receipts]:
        raise ValueError('Code 5 input membership differs from the directory receipt')
    affine = directory['affine_convention']
    rows = []
    for receipt in receipts:
        frame = receipt['frame']
        pair = words[frame]
        data = pair[0][0] + pair[1][0]
        mask = np.concatenate((pair[0][1], pair[1][1]))
        length = data[0]
        span = int.from_bytes(data[5:7], 'little') >> 4
        crc = crc_bytes(data[:length - 3], directory['crc_polynomial_including_top_bit'], affine['seed'], affine['residue'])
        if ((length, span) != (receipt['directory_length'], receipt['decoded_record_span']) or
                not receipt['crc_pass'] or crc != data[length - 3:length] or crc.hex() != receipt['expected_crc'] or
                span not in (135, 163) or int(mask[:8 * length].sum()) != receipt['observed_directory_bits'] or
                not mask[0:8].all() or not mask[40:56].all()):
            raise ValueError(f'Unverified Code 5 directory or layout in frame {frame}')
        record = data[length:length + span]
        end = length + span
        tail = record[-TAIL_BYTES:]
        timing_start = end - TAIL_BYTES + TIMING_BYTE
        word = int.from_bytes(data[timing_start:timing_start + 2], 'little')
        time_bits = mask[8 * timing_start:8 * (timing_start + 2)]
        count_start = 8 * length + RELATIVE_BIT
        selected = record[66:span - TAIL_BYTES]
        basic = structure(data)
        reversed_order = optional_structure(data, basic)['reverse_tag_variant']
        rows.append({'frame': frame, 'group': receipt.get('group'), 'calibration': receipt.get('calibration'),
                     'length': length, 'span': span, 'counter': counter_at(data[:128], width=14),
                     'counter_observed': bool(mask[count_start:count_start + 14].all()),
                     'word': word, 'word_observed': bool(time_bits.all()), **decode_timing_word(word),
                     'selector_65': record[65], 'selector_observed': bool(mask[8 * (length + 65):8 * (length + 66)].all()),
                     'optional_ids': [int.from_bytes(selected[i + 2:i + 4], 'little') for i in range(0, len(selected), 7)],
                     'optional_bytes': selected.hex(),
                     'optional_observed': bool(mask[8 * (length + 66):8 * (end - TAIL_BYTES)].all()),
                     'optional_long_tags_valid': basic['full_match'] or reversed_order if span == 163 else None,
                     'tail_prefix': tail[:34].hex(), 'tail_prefix_observed': bool(mask[8 * (end - TAIL_BYTES):8 * timing_start].all()),
                     'following_marker': data[end:end + 2].hex(), 'following_marker_observed': bool(mask[8 * end:8 * (end + 2)].all()),
                     'record_bit_963': (record[120] >> 3) & 1,
                     'record_bit_963_observed': bool(mask[8 * (length + 120) + 3]),
                     'aligned_tail_byte_for_bit_963': 120 - (span - TAIL_BYTES),
                     'tail_slots': tail[36:44].hex(), 'zero_tail': not any(tail[44:]),
                     'zero_tail_observed': bool(mask[8 * (end - TAIL_BYTES + 44):8 * end].all())})
    return rows


def active_rows(rows):
    return [r for r in rows if r['word'] and r['word_observed'] and r['counter_observed']]


def fit_qin(rows):
    calibration = [r for r in active_rows(rows) if r['calibration']]
    reload = max(r['countdown'] for r in calibration)
    regular = [r for r in calibration if r['countdown'] < reload]
    candidates = []
    for period in range(45, 62):
        votes = Counter((r['counter'] + r['countdown']) % period for r in regular)
        phase, score = votes.most_common(1)[0]
        candidates.append({'period': period, 'phase': phase, 'ordinary_training_matches': score})
    candidates.sort(key=lambda c: -c['ordinary_training_matches'])
    best, runner_up = candidates[:2]
    if best['ordinary_training_matches'] == runner_up['ordinary_training_matches'] or best['period'] != reload:
        raise ValueError('Archive does not identify a unique period consistent with the reload')
    reload_votes = Counter(r['counter'] % best['period'] for r in calibration if r['countdown'] == reload)
    if len(reload_votes) != 1:
        raise ValueError('Calibration does not identify a unique reload phase')
    return {**best, 'reload_phase': next(iter(reload_votes)), 'calibration_candidates': candidates,
            'training_frames': [r['frame'] for r in calibration]}


def score_timing(rows, period, phase, reload):
    selected = active_rows(rows)
    failures = [r['frame'] for r in selected if r['countdown'] != countdown(r['counter'], period, phase, reload)]
    return {'frames': [r['frame'] for r in selected], 'matches': len(selected) - len(failures),
            'failures': failures, 'reload_count': sum(r['countdown'] == period for r in selected),
            'phase': phase, 'reload_phase': reload}


def coverage(windows, layouts, generator):
    code = np.vstack((np.eye(SYSTEM_BITS, dtype=np.uint8), generator.T))
    code = [int.from_bytes(v.tobytes(), 'little') for v in np.packbits(code, axis=1, bitorder='little')]
    groups = []
    for source in windows:
        row = {'kind': source['kind'], 'labels': source['labels'], 'layouts': []}
        for length, span in layouts:
            full = tail_resource_positions(length, span, 0, 16)
            word = (full[0] - CODE5_START) // CODE5_LENGTH
            start = CODE5_START + word * CODE5_LENGTH
            positions = set(visible_positions(start, CODE5_LENGTH, source['carriers']))
            current = {'length': length, 'span': span, 'fields': {}}
            for field, begin, width in (('countdown', 1, 6), ('groups', 9, 3)):
                coords = full[begin:begin + width]
                local = [p - start for p in coords]
                target = interval_bits(local[0], local[-1] + 1)
                current['fields'][field] = {'resources': coords,
                    'carriers': [int(NONPILOT[p % len(NONPILOT)]) for p in coords],
                    'observed_field_bits': [i for i, p in enumerate(local) if p in positions],
                    'ideal_rank': field_information([code[p] for p in sorted(positions)], (), target)}
            row['layouts'].append(current)
        groups.append(row)
    return groups


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    legacy = 'v010-code5-parity-02'
    ut = 'v022-ut-replay'
    observed_480 = artifact(legacy, 'E012', 'code5_observations.npz')
    with np.load(observed_480, allow_pickle=False) as source:
        generator = source['parity_generator']
    model = read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(SYSTEM_BITS), pn_bits(), model)
    directory = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    qin = read_records(archive_words(artifact, pn, generator, directory), directory['archive'], directory)
    raw = {}
    for label, run, exp, npz, report, key in (
            ('480MSps', legacy, 'E012', 'code5_observations.npz', 'code5_parity.json', 'hard_bits'),
            ('250MSps', ut, 'E014', 'ut_observations.npz', 'ut_transfer.json', 'hard_code5_words')):
        words = raw_record_set(artifact(run, exp, npz), read(artifact(run, exp, report))['words'], key, pn, generator)
        raw[label] = read_records(words, directory['raw'][label], directory)
    fit = fit_qin(qin)
    evaluation = [r for r in qin if not r['calibration']]
    period, phase, reload = fit['period'], fit['phase'], fit['reload_phase']
    qin_test = score_timing(evaluation, period, phase, reload)
    for row in fit['calibration_candidates']:
        active_ordinary = [r for r in active_rows(evaluation) if r['countdown'] < period]
        row['ordinary_holdout_matches'] = sum((r['counter'] + r['countdown']) % row['period'] == row['phase']
                                              for r in active_ordinary)
    independent = active_rows(raw['480MSps'])
    first = independent[0]
    raw_phase = (first['counter'] + first['countdown']) % period
    raw_tests = {'480MSps': {'phase_calibration_frame': first['frame'],
                            **score_timing(independent[1:], period, raw_phase, None)},
                 '250MSps': score_timing(raw['250MSps'], period, phase, reload)}
    windows = read(artifact('v077-leo-code5-projection', 'E079', 'leo_code5_projection.json'))['windows']
    layouts = sorted({(r['length'], r['span']) for r in qin})
    decoded = {'Qin': qin, **raw}
    summary = {}
    for label, rows in decoded.items():
        usable = active_rows(rows)
        observed_word = [r for r in rows if r['word_observed']]
        by_span = defaultdict(Counter)
        for row in rows:
            if row['selector_observed'] and row['record_bit_963_observed']:
                by_span[str(row['span'])][f"{row['selector_65']:02x}/{row['record_bit_963']}"] += 1
        summary[label] = {'frames': len(rows), 'timing_words_observed': len(observed_word), 'active_with_counter': len(usable),
                         'inactive_zero_words': sum(r['word'] == 0 for r in observed_word),
                         'nonzero_reserved_bits': sum(bool(r['other_bits']) for r in observed_word),
                         'group_matches': sum(r['groups'] == (r['countdown'] + 7) // 8 for r in observed_word),
                         'observed_selector65_and_bit963': {k: dict(v) for k,v in by_span.items()},
                         'optional_complete_163': sum(r['span'] == 163 and r['optional_observed'] for r in rows),
                         'optional_long_tags_valid_and_observed': sum(bool(r['optional_long_tags_valid']) and r['optional_observed']
                                                                       for r in rows),
                         'tail_prefix_values_direct': sorted({r['tail_prefix'] for r in rows if r['tail_prefix_observed']}),
                         'zero_tail_observed': sum(r['zero_tail_observed'] and r['zero_tail'] for r in rows)}
    result = {'schema': 'starlink-code5-record-countdown/v1',
              'input_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              'qin_model': fit, 'qin_holdout': qin_test, 'raw_tests': raw_tests,
              'summary': summary, 'captures': decoded, 'leo_coverage': coverage(windows, layouts, generator)}
    print('Qin period, phase, reload phase:', period, phase, reload)
    print('Qin holdout:', qin_test['matches'], '/', len(qin_test['frames']), 'reloads', qin_test['reload_count'])
    print('Raw tests:', {key: (value['matches'], len(value['frames'])) for key, value in raw_tests.items()})
    print('Count and packing summaries:', summary)
    output = experiment_output('E137')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'code5_record_countdown.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
