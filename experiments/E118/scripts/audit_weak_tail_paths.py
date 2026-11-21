"""Apply frozen terminal phase to weak-grid paths before reading their CRC."""

from functools import cache
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_offset_constrained_h2 import coded_bits
from audit_ut_repeated_data import checked_artifact, read
from decode_ut_offset_body_first import CRC_RESOURCES, check_crc_tail, header_candidates, load_soft
from layout_core import NONPILOT


PERIOD = 60
FIRST_BODY_BLOCKS = 2
LAST_BODY_BLOCKS = 7
ONSET_SPREAD = 2
SLOT_SLOP = 2
WRONG_PHASE_STEPS = (-8, -2, -1, 1, 2, 8, 30)
PREFIX_STOP = 48
MODES = ('joint', 'prefix_first')


def endpoint(path, h2, widths):
    blocks, sequence = path
    return h2 + blocks*114 + sum(widths[k]*count for k, count in sequence)


def measured_layouts(paths):
    return {(blocks, tuple(tuple(value) for value in seq)) for blocks, seq in paths}


def onset_range(row):
    if row['accepted_phase'] is None:
        return None
    starts = [a['onset']['first_periodic_slot'] for fit in row['fits'] for a in fit]
    if max(starts)-min(starts) > ONSET_SPREAD:
        return None
    length = len(NONPILOT)
    return (max(0, min(starts)-2-SLOT_SLOP)*length, min(300, max(starts)-2+SLOT_SLOP)*length)


def collect_candidates(body_paths, h2, widths, interval, phase):
    lo, hi = interval
    edges = np.arange(lo+(phase-lo) % PERIOD, hi+1, PERIOD, dtype=int)
    stems = list(measured_layouts(body_paths))
    stems.extend((blocks, ()) for blocks in range(FIRST_BODY_BLOCKS, LAST_BODY_BLOCKS+1))
    result = set()
    for stem in stems:
        blocks, seq = stem
        start = endpoint(stem, h2, widths)
        if seq and lo <= start <= hi and start % PERIOD == phase:
            result.add(stem)
        if len(seq) >= min(15, (32*blocks-24)//20):
            continue
        for e in edges[edges > start]:
            for code, width in widths.items():
                quantity, slack = divmod(int(e)-start, width)
                if not slack and 0 < quantity < 4096:
                    result.add((blocks, seq + ((code, quantity),)))
    return sorted(result)


@cache
def modeled_header(layout):
    return [(bits, 1-2*coded_bits(bits).astype(float)) for bits in header_candidates(layout)]


def eligible_words(paths):
    return [(p, bits, signs) for p in paths for bits, signs in modeled_header(p)]


def scores(soft, words, start, end):
    reference = np.stack([signs[start:end] for _, _, signs in words])
    y = soft[start:end].real
    denominator = np.linalg.norm(y) * np.sqrt(end-start)
    return (reference @ y)/max(denominator, 1e-30)


def unique_best(values):
    index = int(values.argmax())
    return index if np.count_nonzero(np.isclose(values, values[index], rtol=1e-10, atol=1e-12)) == 1 else None


def rank(prepared, soft, h2, widths, mode):
    if not prepared:
        return {'candidate_count': 0, 'selected': None}
    words = prepared
    gate = None
    if mode == 'prefix_first':
        prefixes = {}
        for w in words:
            prefixes.setdefault(w[1][:16].tobytes(), w)
        choices = list(prefixes)
        values = scores(soft, list(prefixes.values()), 24, PREFIX_STOP)
        i = unique_best(values)
        if i is None:
            return {'candidate_count': len(words), 'selected': None, 'prefix_ambiguous': True}
        gate = float(values[i])
        words = [w for w in words if w[1][:16].tobytes() == choices[i]]
    stop = min(len(s) for _, _, s in words)-CRC_RESOURCES
    start = PREFIX_STOP if mode == 'prefix_first' else 24
    metric = scores(soft, words, start, stop)
    best = unique_best(metric)
    selected = None
    if best is not None:
        layout, bits, _ = words[best]
        selected = {'path': layout, 'endpoint': endpoint(layout, h2, widths), 'pre_crc_score': float(metric[best]),
                    'pre_crc_start': start, 'pre_crc_stop': stop, 'known_header_prefix_score': gate,
                    'gap_to_next': float(metric[best] - np.delete(metric, best).max()) if len(metric) > 1 else None,
                    **check_crc_tail(soft, bits)}
    return {'candidate_count': len(prepared), 'ranked_count': len(words), 'selected': selected}


def matched_word(first, second):
    a, b = first['selected'], second['selected']
    return a['words_hex'] if a and b and a['crc_ok'] and b['crc_ok'] and a['words_hex'] == b['words_hex'] else None


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    raw, repeat = [doc('v116-ut-weak-tail-phase-final'+suffix, 'E118', 'ut_weak_tail_phase.json')
                   for suffix in ('', '-confirm')]
    if raw != repeat:
        raise ValueError('Weak-grid tail detection results differ')
    a, b = [doc('v108-ut-offset-body-markers'+suffix, 'E110', 'ut_offset_body_markers.json')
            for suffix in ('', '-confirm')]
    if a != b:
        raise ValueError('Body-only path results differ')
    strong_widths = doc('v065-full-tail-phase-final', 'E067', 'tail_phase_audit.json')['resource_widths']
    widths = {int(k): int(w) for k, w in strong_widths.items()}
    h2 = a['h2_start']
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    with np.load(artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_soft.npz'), allow_pickle=False) as z:
        primary, alternate = load_soft(z, config, h2)
    known, check = [doc('v108-ut-offset-body-first'+suffix, 'E110', 'ut_offset_body_first.json')
                    for suffix in ('', '-confirm')]
    if known != check:
        raise ValueError('Prior H2 reads differ')
    by_tick = {r['tick']: r for r in known['rows']}
    paths = {r['tick']: r['body_paths'] for r in a['rows']}
    rows = []
    for row in raw['weak']:
        tick, phase = row['tick'], row['accepted_phase']
        interval = onset_range(row)
        record = {'tick': tick, 'phase': phase, 'coarse_endpoint_interval': interval}
        if phase is not None:
            supported = [a['onset'] for fit in row['fits'] for a in fit]
            record['onset_min_held_median_improvement'] = min(
                a['post_onset_held_median_before_phase_search']-a['pre_onset_held_median'] for a in supported)
        prior = by_tick[tick]
        explicit_prior = matched_word(prior['primary'], prior['other_pilot_fit'])
        original = prior['primary']['selected'] if explicit_prior else None
        if original:
            layout = (len(original['words_hex']), [(e['mcs_candidate'], e['quantity'])
                                                    for e in original['layout']['entries']])
            stop = endpoint(layout, h2, widths)
            record['prior_validated_h2'] = {'words_hex': explicit_prior, 'endpoint': stop, 'phase_matches': stop % PERIOD == phase,
                                            'inside_interval': interval is not None and interval[0] <= stop <= interval[1]}
        if interval is None:
            rows.append(record)
            continue
        perturbations = []
        measured = measured_layouts(paths[tick])
        for shift in (0, *WRONG_PHASE_STEPS):
            tested_phase = (phase+shift) % PERIOD
            candidates = collect_candidates(paths[tick], h2, widths, interval, tested_phase)
            words = eligible_words(candidates)
            predictions = {}
            for mode in MODES:
                x = rank(words, primary[tick], h2, widths, mode)
                y = rank(words, alternate[tick], h2, widths, mode)
                predictions[mode] = {'primary': x, 'alternate': y, 'matching_valid_words': matched_word(x, y)}
            # Apply the same terminal constraint without introducing a virtual last marker.
            directly_observed = eligible_words([p for p in candidates if p in measured])
            x = rank(directly_observed, primary[tick], h2, widths, 'joint')
            y = rank(directly_observed, alternate[tick], h2, widths, 'joint')
            predictions['direct_markers'] = {'primary': x, 'alternate': y, 'matching_valid_words': matched_word(x, y)}
            perturbations.append({'phase_shift': shift, 'phase': tested_phase, 'paths': candidates,
                                  'predictions': predictions})
        record['phase_arms'] = perturbations
        rows.append(record)
        print(tick, phase, interval, len(perturbations[0]['paths']),
              {m: d['matching_valid_words'] for m, d in perturbations[0]['predictions'].items()}, flush=True)
    result = {'schema': 'starlink-ut-weak-tail-paths/v2', 'inputs_sha256': {p.as_posix(): d for p, d in hashes.items()},
              'mcs_widths': widths, 'h2_start': h2, 'wrong_phase_steps': WRONG_PHASE_STEPS, 'rows': rows}
    destination = experiment_output('E118')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'ut_weak_tail_paths.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n',
                                                         encoding='utf-8')


if __name__ == '__main__':
    main()
