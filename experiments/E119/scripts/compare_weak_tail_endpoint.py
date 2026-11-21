"""Replay H2 with frozen resource endpoint intervals before checking CRC."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_weak_tail_paths import MODES, eligible_words, endpoint, matched_word, measured_layouts, rank
from decode_ut_offset_body_first import load_soft


CONTROL_STEPS = (-1200, -600, 600, 1200)


def restrict(paths, bounds, h2, widths):
    lo, hi = bounds
    return [p for p in paths if lo <= endpoint(p, h2, widths) <= hi]


def evaluate(paths, markers, soft, h2, widths):
    words = eligible_words(paths)
    direct = eligible_words([p for p in paths if p in markers])
    rows = {}
    for method, candidates in [(mode, words) for mode in MODES] + [('direct_markers', direct)]:
        mode = method if method in MODES else 'joint'
        a, b = [rank(candidates, received, h2, widths, mode) for received in soft]
        rows[method] = {'primary': a, 'other_pilot_fit': b, 'common_crc_valid': matched_word(a, b)}
    return rows


def main():
    require_run_context()
    hashes = {}

    def doc(run, exp, name):
        return read(checked_artifact(run, exp, name, hashes))

    received, check = [doc('v117-ut-weak-resource-end-final'+s, 'E119', 'ut_weak_tail_endpoint.json')
                       for s in ('', '-confirm')]
    if received != check:
        raise ValueError('Frozen endpoint fits disagree')
    coarse = doc('v116-ut-weak-tail-paths-markers', 'E118', 'ut_weak_tail_paths.json')
    body = doc('v108-ut-offset-body-markers', 'E110', 'ut_offset_body_markers.json')
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    h2 = coarse['h2_start']
    widths = {int(k): v for k, v in coarse['mcs_widths'].items()}
    with np.load(checked_artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_soft.npz', hashes),
                 allow_pickle=False) as data:
        primary, alternate = load_soft(data, config, h2)
    older = {row['tick']: row for row in coarse['rows']}
    marked = {row['tick']: measured_layouts(row['body_paths']) for row in body['rows']}
    results = []
    for r in received['rows']:
        tick = r['tick']
        lohi = r['retained_interval']
        row = {'tick': tick, 'refined_interval': lohi, 'phase': r['phase']}
        if lohi is not None:
            origin = older[tick]
            if origin['phase'] != r['phase'] or origin['coarse_endpoint_interval'] != r['coarse_interval']:
                raise ValueError('The phase or coarse extent disagrees between experiments')
            candidates = sorted(measured_layouts(origin['phase_arms'][0]['paths']))
            chosen = restrict(candidates, lohi, h2, widths)
            soft = (primary[tick], alternate[tick])
            attempt = evaluate(chosen, marked[tick], soft, h2, widths)
            report = []
            for shift in CONTROL_STEPS:
                left, right = lohi[0]+shift, lohi[1]+shift
                if left < r['coarse_interval'][0] or right > r['coarse_interval'][1]:
                    continue
                wrong = restrict(candidates, (left, right), h2, widths)
                report.append({'shift': shift, 'interval': [left, right], 'paths': wrong,
                               'results': evaluate(wrong, marked[tick], soft, h2, widths)})
            row.update(coarse_path_count=len(candidates), retained_paths=chosen, results=attempt,
                       translated_controls=report)
            print(tick, len(candidates), len(chosen), {m: d['common_crc_valid'] for m, d in attempt.items()}, flush=True)
        results.append(row)

    truth = {}
    for row in coarse['rows']:
        if 'prior_validated_h2' in row:
            truth[row['tick']] = row['prior_validated_h2']
    for row in results:
        tick = row['tick']
        if tick not in truth:
            continue
        ends = row['refined_interval']
        expected = truth[tick]
        row['prior_endpoint'] = expected['endpoint']
        row['prior_endpoint_in_refined_interval'] = ends is not None and ends[0] <= expected['endpoint'] <= ends[1]
        row['prior_word_reproduced'] = {m: d['common_crc_valid'] == expected['words_hex']
                                       for m, d in row.get('results', {}).items()}

    output = {'schema': 'starlink-weak-tail-h2-constraint/v1', 'translated_offsets': CONTROL_STEPS,
              'input_sha256': {p.as_posix(): h for p, h in hashes.items()}, 'rows': results}
    out = experiment_output('E119')
    out.mkdir(parents=True, exist_ok=True)
    (out/'ut_weak_tail_endpoint_h2.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n',
                                                     encoding='utf-8')


if __name__ == '__main__':
    main()
