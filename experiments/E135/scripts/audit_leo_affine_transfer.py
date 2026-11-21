"""Test fixed LEO early-symbol XOR relations against verified wideband captures."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_leo_copy_transfer import file_hash
from audit_leo_early_symbol_roles import CODE5_LENGTH, CODE5_START, REFERENCE
from audit_leo_lower_edge_relation import resource_coordinates, signed_bin
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_reference


STUDY = Path('reports/2026_09_28_sequence_semantics')


def resource_role(resource):
    if resource < CODE5_START:
        return {'part': 'H1', 'offset': resource}
    word, offset = divmod(resource - CODE5_START, CODE5_LENGTH)
    part = 'Code 5 systematic' if offset < CODE5_LENGTH // 2 else 'Code 5 parity'
    return {'part': part, 'word': word, 'offset': offset}


def relation_traces(bits, valid, left, right, constant):
    def xor(columns, mask_bit):
        selected = bits[:, list(columns)]
        values = np.bitwise_xor.reduce(selected, axis=1) ^ mask_bit
        return values, valid[:, list(columns)].all(axis=1)

    def masked(values, mask):
        return ''.join(str(int(value)) if keep else '?' for value, keep in zip(values, mask, strict=True))

    a, a_valid = xor(left, 0)
    b, b_valid = xor(right, constant)
    return {'left': masked(a, a_valid), 'adjusted_right': masked(b, b_valid),
            'joint': masked(a == b, a_valid & b_valid)}


def counts(trace):
    qualified = len(trace['joint']) - trace['joint'].count('?')
    left_joint = {a for a, result in zip(trace['left'], trace['joint'], strict=True) if result != '?'}
    return {'qualified': qualified, 'matches': trace['joint'].count('1'),
            'erasures': trace['joint'].count('?'), 'varying_left': len(left_joint) == 2}


def source_relations():
    names = ('HEADER_BLOCK_RANK.md', 'header_parity_translation.py', 'shared_header_coordinates.py',
             'local/header_parity_translation.json', 'local/rectangle_rank_control.json',
             'local/shared_header_coordinates.json')
    paths = {name: EXTERNAL_LEO_ROOT / STUDY / name for name in names}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    translation, control, shared = [read(paths[name]) for name in names[-3:]]
    if next(value for path, value in shared['input_sha256'].items()
            if path.endswith('/rectangle_rank_control.json')) != hashes[names[-2]]:
        raise ValueError('LEO shared-coordinate inputs changed')
    windows = {(tuple(r['symbols']), r['start_bin']) for r in control['rows'] if not r['shifted_at_most32']}
    if windows != {(tuple(r['symbols']), r['start_bin']) for r in shared['windows']} or len(windows) != 37:
        raise ValueError('LEO rectangle selection differs from the frozen receipt')
    original = [r for r in translation['rows'] if r['delta'] == 0]
    if (len(original) != 2 or len(shared['coordinates']) != 70 or
            any(r['discovery_count'] != 39 or r['evaluation_count'] != 39 or r['raw_count'] != 7 or
                r['discovery_errors'] or r['evaluation_errors'] or r['raw_errors'] for r in original)):
        raise ValueError('LEO relation inventory changed')
    # Both zero-translation relations explicitly use OFDM symbol 2 in the source report.
    relations = [{'name': f"nine-{r['candidate']}", 'left': [(2, k) for k in r['bins']],
                  'right': [], 'constant': r['parity']} for r in original]
    relations.extend({'name': f'shared-{i:02d}', 'left': r['left'], 'right': r['right'],
                      'constant': r['constant'], 'reference_left': r['reference_bits']}
                     for i, r in enumerate(shared['coordinates']))
    return relations, hashes, shared['windows'], shared['input_sha256']


def qin_signs(artifact, coords, rotation, resources):
    run = 'v032-qin-snapshot'
    headers = {r['frame']: r for r in read(artifact('v034-qin-header-survey', 'E036',
                                                   'qin_header_survey.json'))['frames']}
    groups = read(artifact(run, 'E034', 'qin_snapshot_source.json'))['hard_groups']
    symbols, bins = np.array(coords).T
    rows = []
    for group in groups:
        with np.load(artifact(run, 'E034', f'qin_hard_{group}.npz'), allow_pickle=False) as source:
            for frame, values in zip(source['frame'], source['symbols'], strict=True):
                header = headers[int(frame)]
                h1 = header['h1']
                first = h1['layout20']['entries'][0]
                if (not h1['crc_ok'] or not header['h2']['crc_ok'] or h1['needed_resources'] != CODE5_START
                        or first['mcs_candidate'] != 5 or first['quantity'] < 2
                        or header['h2_start'] <= max(resources)):
                    raise ValueError(f'Unverified Code 5 allocation in Qin frame {frame}')
                signs = values[symbols - 1, bins] * rotation
                valid = (abs(signs.imag) < 0.05) & (abs(abs(signs.real) - 1) < 0.05)
                rows.append((int(frame), signs.real >= 0, valid, group))
    rows.sort(key=lambda r: r[0])
    if len(rows) != len({r[0] for r in rows}):
        raise ValueError('Duplicate Qin frames')
    return {'frame': [r[0] for r in rows], 'bits': np.stack([r[1] for r in rows]),
            'mask': np.stack([r[2] for r in rows]), 'block': [r[3] for r in rows]}


def independent_signs(artifact, resources, ratios):
    body = 'v010-code5-parity-02'
    h1 = 'v008-complete-header-layout-02'
    index = read(artifact(body, 'E012', 'code5_parity.json'))
    headers = read(artifact(h1, 'E010', 'full_header_layout.json'))['frames']
    key = lambda row: row['legacy_frame_index']
    h1_disagreements = {r['frame']: r['h1_known_hard_disagreements'] for r in index['receiver']}
    bodies = {}
    with np.load(artifact(body, 'E012', 'code5_observations.npz'), allow_pickle=False) as source:
        generator = source['parity_generator']
        for row, word in zip(index['words'], source['hard_bits'], strict=True):
            number = row['word']
            if row['field'] != 'first' or number not in (0, 1):
                continue
            if (row['start_resource'] != CODE5_START + number * CODE5_LENGTH or
                    np.any((word[:1024] @ generator) % 2 != word[1024:])):
                raise ValueError('Wideband Code 5 word fails its original checks')
            frame = bodies.setdefault(row['frame'], {})
            if number in frame:
                raise ValueError('Duplicate wideband Code 5 word')
            frame[number] = word
    bits, frames = [], []
    with np.load(artifact(h1, 'E010', 'full_header_observations.npz'), allow_pickle=False) as source:
        for row in sorted(headers, key=key):
            frame = key(row)
            header = row['first_header']
            prefix = f"frame_{row['capture_ordinal']:02d}_h1_"
            h1_bits = source[prefix + 'reencoded']
            # E010 emits its H1 soft signs with bit 1 on the positive real axis.
            # The E012 analytical receiver verifies the same encoded bits on the negative axis.
            if (not header['crc_ok'] or header['coded_hard_disagreements'] or h1_disagreements[frame]
                    or not np.array_equal(h1_bits, source[prefix + 'soft'] > 0)
                    or row['first_layout20']['entries'][0]['mcs_candidate'] != 5
                    or row['first_layout20']['entries'][0]['quantity'] < 2):
                raise ValueError('Wideband H1 or the Code 5 allocation fails its original checks')
            full = np.concatenate((h1_bits, bodies[frame][0], bodies[frame][1]))
            bits.append((1 - 2 * full[resources].astype(float)) * ratios.real >= 0)
            frames.append(frame)
    bits = np.stack(bits)
    return {'frame': frames, 'bits': bits, 'mask': np.ones_like(bits)}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    relations, source_hashes, windows, underlying = source_relations()
    coordinates = sorted({tuple(c) for r in relations for c in r['left'] + r['right']})
    resources = resource_coordinates(coordinates)
    coord_indices = {c: i for i, c in enumerate(coordinates)}
    roles = [resource_role(r) for r in resources]
    syms, bins = np.array(coordinates).T
    inputs[REFERENCE] = file_hash(REFERENCE)
    if inputs[REFERENCE] != next(value for path, value in underlying.items() if path.endswith('/referenceTemplate.mat')):
        raise ValueError('Qin phase reference differs from the one used by LEO')
    template = loadmat(REFERENCE)['referenceTemplateRotations']
    rotation = np.exp(-0.5j * np.pi * template[bins, syms - 1])
    ratios = rotation * np.array([pn_reference(signed_bin(k), np.array([s]))[0] for s, k in coordinates])
    if np.any(abs(ratios.imag) > 1e-10):
        raise ValueError('Phase conversion is not real at every fixed coordinate')
    qin = qin_signs(artifact, coordinates, rotation, resources)
    raw = independent_signs(artifact, resources, ratios)
    qin_frames = np.array(qin['frame'])
    discovery_frames = [frame for frame in qin['frame'] if frame < 78]
    groups = {'Qin_LEO_0_77': (qin, qin_frames < 78), 'Qin_after_77': (qin, qin_frames >= 78),
              'independent_480MSps': (raw, np.ones(len(raw['frame']), dtype=bool))}

    for relation in relations:
        left = tuple(coord_indices[tuple(c)] for c in relation['left'])
        right = tuple(coord_indices[tuple(c)] for c in relation['right'])
        relation['resources'] = [resources[i] for i in left + right]
        relation['roles'] = [roles[i] for i in left + right]
        relation['results'] = {}
        for name, (dataset, rows) in groups.items():
            trace = relation_traces(dataset['bits'][rows], dataset['mask'][rows], left, right, relation['constant'])
            relation['results'][name] = {**trace, **counts(trace)}
        source = relation['results']['Qin_LEO_0_77']
        if not discovery_frames or source['matches'] != source['qualified'] or source['qualified'] != len(discovery_frames):
            raise ValueError(f"Current Qin phase/coordinate convention does not replay {relation['name']}")
        if 'reference_left' in relation:
            kept = [relation['reference_left'][frame] for frame in discovery_frames]
            if source['left'] != ''.join(kept):
                raise ValueError(f"Original Qin left trace changed for {relation['name']}")

    def group_summary(group, candidates):
        qualified = [r['results'][group] for r in candidates if r['results'][group]['qualified']]
        return {'evaluated': len(qualified),
                'exact': sum(r['matches'] == r['qualified'] for r in qualified),
                'exact_with_varying_left': sum(r['matches'] == r['qualified'] and r['varying_left'] for r in qualified)}

    summary = {group: {'nine': group_summary(group, relations[:2]),
                       'shared': group_summary(group, relations[2:])} for group in groups}
    for group, report in summary.items():
        print(group, report)
    print('Fixed source coordinates by role:', dict(Counter(r['part'] for r in roles)))
    result = {'schema': 'starlink-leo-affine-transfer/v1', 'leo_inputs_sha256': source_hashes,
              'leo_underlying_receipt_sha256': underlying,
              'project_inputs_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): digest for p, digest in inputs.items()},
              'leo_windows': windows, 'summary': summary,
              'frame_groups': {name: [frame for frame, keep in zip(d['frame'], rows, strict=True) if keep]
                               for name, (d, rows) in groups.items()},
              'qin_archive_blocks': dict(zip(qin['frame'], qin['block'], strict=True)),
              'coordinate_phase_signs': [int(v) for v in ratios.real.round()], 'coordinates': coordinates,
              'relations': relations}
    destination = experiment_output('E135')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'leo_affine_transfer.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
