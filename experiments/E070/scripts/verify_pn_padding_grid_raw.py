"""Test the archive-derived byte grid on retained soft symbols from the raw capture."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'learning')]
from project_paths import experiment_output, require_run_context
from audit_pn_padding_starts import GRIDS, MARKERS, prefix_profile, word_windows
from audit_qam_pn15 import equality_predictions
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_qam_fec import slice_labels
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


MARGINS = (0., .01)


def audit_words(observations, constellation, mapping, coords, pn, system_end):
    decoded, confidence = slice_labels(observations, constellation, mapping)
    bits = decoded.reshape(len(observations), -1)
    outputs = []
    for margin in MARGINS:
        known = np.broadcast_to((confidence >= margin)[..., None], decoded.shape).reshape(bits.shape)
        state = word_windows(bits, known, pn, system_end)
        rows = []
        for i in np.flatnonzero(state['passed']):
            frame, start, final = coords[i]
            r = prefix_profile(bits[i], known[i], pn, system_end)
            r.update(frame=frame, resource_start=start, last=final)
            rows.append(r)
        outputs.append({'margin': margin, 'words': len(bits),
                        'eligible_terminal': sum(bool(x) and c[2] for x, c in zip(state['available'], coords)),
                        'eligible_nonterminal': sum(bool(x) and not c[2] for x, c in zip(state['available'], coords)),
                        'strong_words': rows})
    return outputs


def main():
    require_run_context()
    inputs = {}
    def checked(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    archive = read(checked('v068-pn-padding-starts', 'E070', 'pn_padding_starts.json'))
    e028 = checked('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz')
    e030 = checked('v028-ut-cross32-extract-02', 'E030', 'ut_cross32_symbols.npz')
    crosses = read(checked('v028-ut-cross32-extract-02', 'E030', 'ut_cross32_extraction.json'))
    labels = read(checked('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    capture = read(checked('v022-ut-replay', 'E014', 'ut_transfer.json'))
    inventory = read(checked('v066-descriptor-layout-final', 'E068', 'descriptor_layout_identifiability.json'))
    known = next(c for c in inventory['candidates'] if c['layout'] == inventory['historical_h2_layout'])
    widths = dict(zip(known['labels'], known['absolute']['units'], strict=True))
    regions = {}
    for f in capture['candidates']:
        if not f['complete_frame']:
            continue
        h = f['h2']
        current = f['predicted_h2_start']+114*h['blocks']
        for e in h['layout20']['entries']:
            c, n = e['mcs_candidate'], e['quantity']
            regions[f['candidate_index'], c] = current, n
            current += n*widths[c]
    p = equality_predictions(np.arange(16385), pn_bits(), labels['frozen_pn_model'])
    by_code = {}
    with np.load(e028, allow_pickle=False) as data, np.load(e030, allow_pickle=False) as data32:
        const16, map16 = data['constellation_points'], data['frozen_gray_mapping']
        const32 = np.asarray([complex(r['i'], r['q']) for r in labels['cross32_mapping']])/np.sqrt(20)
        map32 = labels_as_bits(np.array([r['label'] for r in labels['cross32_mapping']]), 5)
        for c in (57, 66, 70, 71, 78, 85, 88, 92):
            rule = archive['grid_evidence_by_code'][str(c)]
            arm = next(k for k in rule['arms'] if k['grid'] == 8)
            if rule['learned_grid'] != 8 or arm['calibration_selected_remainder'] != 0:
                raise ValueError('Archive grid no longer matches frozen raw test')
            if c in (66, 70):
                values = data[f'mcs{c}_refined'][:, 1:-1]
                keys = zip(data[f'mcs{c}_frames'].tolist(), data[f'mcs{c}_starts'].tolist(), strict=True)
                constellation, mapping = const16, map16
            else:
                values = data32[f'mcs{c}_symbols']
                keys = [(r['frame'], r['start']+k*widths[c]) for r in crosses['regions'] if r['format'] == c
                        for k in range(r['count'])]
                if [f for f, _ in keys] != data32[f'mcs{c}_frames'].tolist():
                    raise ValueError('Raw extracted-word index changed')
                constellation, mapping = (const16, map16) if c == 57 else (const32, map32)
            coords = []
            for f, start in keys:
                origin, count = regions[f, c]
                j, rem = divmod(start-origin, widths[c])
                if rem or not 0 <= j < count:
                    raise ValueError('Raw word does not land on the expected coordinate')
                coords.append((f, start, j == count-1))
            by_code[str(c)] = audit_words(values, constellation, mapping, coords, p, MARKERS[c])
    summary = []
    for i, margin in enumerate(MARGINS):
        groups = defaultdict(list)
        for c, result in by_code.items():
            strongest = result[i]['strong_words']
            for row in strongest:
                if row['grid_log_evidence'] is not None and row['last']:
                    groups[row['frame']].append(row['grid_log_evidence'])
        summary.append({'margin': margin,
                        'mean_evidence_by_grid': [np.mean([np.mean([w[i] for w in words], axis=0)
                                                          for words in groups.values()], axis=0).tolist()
                                                  for i in range(len(GRIDS))],
                        'frames': sorted(groups), 'terminal_words_in_comparison': sum(map(len, groups.values()))})
    out = experiment_output('E070')
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'pn-padding-grid-raw/v1', 'fixed_grid': 8, 'fixed_remainder': 0,
              'grids': GRIDS, 'inputs_sha256': {k.relative_to(ROOT).as_posix(): v for k, v in inputs.items()},
              'per_code': by_code, 'pooled_by_frame': summary}
    (out/'pn_padding_grid_raw.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
