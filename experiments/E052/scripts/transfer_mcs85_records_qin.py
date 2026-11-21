"""Transfer the measured MCS 85 record template to archived Qin symbols."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_mcs85_records import SYSTEMATIC_ENDS, check_frame, detect_records, system_bytes
from audit_qin_lengths import AUDIT, SNAPSHOT, NONPILOT, pdu_stream, regions_for, pn_reference
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, equality_predictions, pn_bits, read
from audit_ut_weak_frames import PREFIX


def main():
    require_run_context()
    inputs = {}

    def artifact(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    fit = artifact('v050-mcs85-spacing', 'E052', 'mcs85_records.json')
    groups = artifact(SNAPSHOT, 'E034', 'qin_snapshot_source.json')['hard_groups']
    paths = [checked_artifact(SNAPSHOT, 'E034', f'qin_hard_{group}.npz', inputs) for group in groups]
    audit = artifact(AUDIT, 'E034', 'qin_exemplar_audit.json')
    lengths = artifact('v033-qin-lengths-final', 'E035', 'qin_lengths.json')
    qam = artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')
    by_frame = {row['frame']: row for row in audit['headers']}
    group_of = {row['frame']: row['group'] for row in lengths['frames']}
    overlapping = {group_of[row['qin_frame']] for row in audit['overlap']['pairs']}
    widths = {int(code): width for key in ('inherited_units', 'fitted_units') for code, width in lengths[key].items()}
    slots = np.arange(2, 302)
    carrier_reference = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    points = np.array([complex(row['i'], row['q']) for row in qam['cross32_mapping']])/np.sqrt(20)
    mapping = labels_as_bits(np.array([row['label'] for row in qam['cross32_mapping']]), 5)
    masks = [np.frombuffer(bytes.fromhex(fit[name+'_hex']), dtype=np.uint8)
             for name in ('template', 'detection_mask', 'verification_mask')]
    reference = fit['pn_model']
    loc = np.arange(16384)
    pn = equality_predictions(loc, pn_bits(), reference)
    wrong_pn = equality_predictions(loc, pn_bits(), {**reference, 'phase': reference['phase']+1})
    distances = fit['fitted_distances_by_exact_label']
    out = []
    for index, group, _, flat in pdu_stream(paths, groups, carrier_reference, PREFIX):
        for code, count, start, stop in regions_for(by_frame[index], widths):
            if code != 85:
                continue
            body = (flat[start:stop]*max(abs(points))).reshape(count, widths[code])
            hard = mapping[nearest_symbols(body, points)].reshape(count, -1)[:, :16384]
            bytes_ = system_bytes(hard, SYSTEMATIC_ENDS[code], pn).ravel()
            wrong = system_bytes(hard, SYSTEMATIC_ENDS[code], wrong_pn).ravel()
            found = detect_records(bytes_, *masks, max_distance=fit['detection_max_errors'])
            control = detect_records(wrong, *masks, max_distance=fit['detection_max_errors'])
            checks, alternatives, steps = check_frame(found, len(bytes_), distances)
            out.append({'frame': index, 'group': group, 'candidate_words': count, 'resource_start': start,
                        'records': found, 'shifted_pn_control_records': control, 'length_predictions': checks,
                        'position_controls': alternatives, 'candidate_count_steps': steps})
    result = {'schema': 'starlink-mcs85-qin-transfer/v1', 'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p,h in inputs.items()},
              'overlap_groups': sorted(overlapping), 'fitted_distances_by_exact_label': distances, 'results': out}
    output = experiment_output('E052')
    output.mkdir(parents=True, exist_ok=True)
    (output/'qin_record_transfer.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, subgroup in (('same waveform group', [r for r in out if r['group'] in overlapping]),
                           ('other measured groups', [r for r in out if r['group'] not in overlapping])):
        print(name, len(subgroup), 'frames;', sum(len(r['records']) for r in subgroup), 'records', flush=True)


if __name__ == '__main__':
    main()
