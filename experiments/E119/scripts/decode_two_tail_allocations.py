"""Use a frozen endpoint to enumerate two-descriptor paths without PN markers."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_weak_tail_paths import MODES, eligible_words, matched_word, rank
from decode_ut_offset_body_first import load_soft


CONTROL_SHIFTS = (-600, -120, -60, 60, 120, 600)


def paths_closing_at(endpoint, h2, widths):
    by_width = defaultdict(list)
    for code, width in sorted(widths.items()):
        by_width[width].append(code)
    result = []
    for blocks in range(2, 8):
        length = endpoint-h2-114*blocks
        for w1, codes1 in sorted(by_width.items()):
            for w2, codes2 in sorted(by_width.items()):
                for n1 in range(1, min(4095, (length-w2)//w1)+1):
                    n2, slack = divmod(length - n1*w1, w2)
                    if slack or not 0 < n2 < 4096:
                        continue
                    result.extend((blocks, ((k1, n1), (k2, n2))) for k1 in codes1 for k2 in codes2)
    return sorted(result)


def main():
    require_run_context()
    hashes = {}

    def doc(run, exp, name):
        return read(checked_artifact(run, exp, name, hashes))

    a, b = [doc('v117-ut-weak-resource-end-final'+suffix, 'E119', 'ut_weak_tail_endpoint.json')
            for suffix in ('', '-confirm')]
    if a != b:
        raise ValueError('Frozen exact-end observations disagree')
    input_record = doc('v116-ut-weak-tail-paths-markers', 'E118', 'ut_weak_tail_paths.json')
    h2 = input_record['h2_start']
    widths = {int(k): v for k, v in input_record['mcs_widths'].items()}
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    path = checked_artifact('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_soft.npz', hashes)
    with np.load(path, allow_pickle=False) as z:
        soft, alternate = load_soft(z, config, h2)
    result = []
    for row in a['rows']:
        if row['retained_interval'] is None or not row['all_arms_same']:
            continue
        tick = row['tick']
        endpoint = row['min_selected']
        tested = []
        for shift in (0, *CONTROL_SHIFTS):
            trial = endpoint+shift
            if not row['coarse_interval'][0] <= trial <= row['coarse_interval'][1]:
                continue
            paths = paths_closing_at(trial, h2, widths)
            candidates = eligible_words(paths)
            methods = {}
            for method in MODES:
                primary = rank(candidates, soft[tick], h2, widths, method)
                secondary = rank(candidates, alternate[tick], h2, widths, method)
                methods[method] = {'primary': primary, 'other_pilot_fit': secondary,
                                   'common_crc_valid': matched_word(primary, secondary)}
            tested.append({'shift': shift, 'tested_endpoint': trial, 'paths': paths, 'methods': methods})
        result.append({'tick': tick, 'endpoint': endpoint, 'trials': tested})
        print(tick, endpoint, len(tested[0]['paths']),
              {method: row['common_crc_valid'] for method, row in tested[0]['methods'].items()}, flush=True)
    known = {row['tick']: row['prior_validated_h2'] for row in input_record['rows'] if 'prior_validated_h2' in row}
    for row in result:
        row['prior_validated_h2'] = known.get(row['tick'])
    output = {'schema': 'starlink-two-allocations-by-tail/v1', 'control_shifts': CONTROL_SHIFTS,
              'input_sha256': {p.as_posix(): h for p, h in hashes.items()}, 'rows': result}
    out = experiment_output('E119')
    out.mkdir(parents=True, exist_ok=True)
    (out/'ut_weak_tail_two_allocations.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n',
                                                        encoding='utf-8')


if __name__ == '__main__':
    main()
