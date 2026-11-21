"""Validate CRC24 directory lengths across independently parsed H2 allocations."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_record_fields import SYSTEMATIC, make_pn, raw_streams
from audit_mcs6_body_spans import span_from_entry
from audit_ut_repeated_data import checked_artifact, read
from extract_ut_qam import allocation_regions
from layout_core import NONPILOT


def leo_direct(directory, resource, descriptors, windows):
    def bins(positions):
        return NONPILOT[(resource + np.asarray(positions) // 4) % len(NONPILOT)]
    length_bins = bins([np.arange(pos * 8 + 20, pos * 8 + 40) for pos in descriptors])
    directory_bins = bins(np.arange(directory * 8))
    return [{"kind": r["kind"], "labels": r["labels"],
             "fully_seen_length_fields": int(np.isin(length_bins, r["carriers"]).all(axis=1).sum()),
             "complete_directory": bool(np.isin(directory_bins, r["carriers"]).all())} for r in windows]


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    receipt = read(artifact('v132-outer-record-lengths', 'E134', 'outer_record_lengths.json'))['wideband']
    headers = read(artifact('v022-ut-replay', 'E014', 'ut_transfer.json'))['candidates']
    stop = read(artifact('v023-binary-tail-final', 'E025', 'ut_binary_tail.json'))['conditional_length_audit']
    units = {int(k): value for k, value in stop['candidate_resource_units'].items()}
    windows = read(artifact('v128-leo-full-code-observability', 'E130', 'leo_full_code_observability.json'))['windows']
    labels = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    sources = {(code, frame, resource): data.tobytes() for code, frame, resource, _, data in
               raw_streams(artifact, labels['frame_split']['reserved'], make_pn(labels['frozen_pn_model']))}
    raw = {(row['mcs'], row['frame'], row['resource']): row for row in receipt}
    output = []
    for item in receipt:
        directory = item['directory']
        if not directory:
            continue
        if item['mcs'] != 66:
            raise ValueError('LEO projection here is fixed for four bits per MCS 66 resource')
        frame = item['frame']
        source = sources[66, frame, item['resource']]
        fmt = directory['partition']
        descriptors = [offset for offset, tag in zip(fmt['entry_offsets'], fmt['entry_tags'], strict=True) if tag == 3]
        predicted = [span_from_entry(source[offset:offset + 5]) for offset in descriptors]
        case = next(row for row in headers if row['candidate_index'] == frame)
        if not case['h2']['crc_ok']:
            raise ValueError('Missing CRC-checked H2 ordering')
        regions = []
        for code, start, end, words in allocation_regions(case, units):
            if start < item['resource']:
                continue
            received = raw.get((code, frame, start))
            if received is not None and received['capacity'] != words * SYSTEMATIC[code] // 8:
                raise ValueError('Allocation capacity disagrees with H2')
            lengths = None if received is None else [r['stop'] - r['start'] for r in received['records']]
            regions.append({'mcs': code, 'resources': [start, end], 'h2_codewords': words, 'lengths_from_outer_headers': lengths})
        decoded = [length for region in regions for length in (region['lengths_from_outer_headers'] or [])]
        if decoded != predicted or regions[0]['resources'][0] != item['resource']:
            raise ValueError('Fixed directory order does not match lengths read inside the waveform')
        controls = {'reverse': list(reversed(decoded)), 'rotated_one': decoded[1:] + decoded[:1]}
        checks = {label: sum(a == b for a, b in zip(predicted, values, strict=True)) for label, values in controls.items()}
        output.append({'frame': frame, 'directory': directory, 'descriptor_offsets': descriptors,
                       'predicted_lengths': predicted, 'h2_order': regions, 'control_matches': checks,
                       'leo_direct_projection': leo_direct(directory['length'], item['resource'], descriptors, windows)})
    folder = experiment_output('E134')
    folder.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'starlink-cross-mcs-directory-links/v1', 'frames': output,
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    (folder / 'cross_mcs_directory_links.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in output:
        print(row['frame'], row['directory']['crc_hex'], len(row['predicted_lengths']), 'exact ordered lengths;',
              'max LEO full fields', max(w['fully_seen_length_fields'] for w in row['leo_direct_projection']), flush=True)


if __name__ == '__main__':
    main()
