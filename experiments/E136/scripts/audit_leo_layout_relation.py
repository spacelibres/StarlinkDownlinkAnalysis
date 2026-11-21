"""Validate LEO's remaining XORs and test dependence on Code 5 record layouts."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, crc_bytes, raw_record_set
from audit_code5_frame_count import descramble
from audit_leo_affine_transfer import counts, relation_traces
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference


CANDIDATES = ('shared-28', 'shared-29', 'shared-48', 'shared-53')


def layout_role(bit, length, span):
    if bit < 8 * (length - 3):
        return 'directory_body'
    if bit < 8 * length:
        return 'directory_crc24'
    if bit < 8 * (length + span):
        return 'record_1'
    return 'following_record'


def template_signs(words, scrambler, signs):
    return words ^ scrambler ^ (signs > 0)


def predict_by_layout(layouts, signs, calibration):
    votes = defaultdict(Counter)
    for layout, sign, train in zip(layouts, signs, calibration, strict=True):
        if train and sign != '?':
            votes[layout][sign] += 1
    output = {}
    for layout, vote in votes.items():
        if vote['0'] != vote['1']:
            output[layout] = str(int(vote['1'] > vote['0']))
    return ''.join(output.get(layout, '?') for layout in layouts)


def score(predictions, observations, include):
    correct = wrong = erased = 0
    for p, y, keep in zip(predictions, observations, include, strict=True):
        if keep:
            if '?' in (p, y):
                erased += 1
            elif p == y:
                correct += 1
            else:
                wrong += 1
    return {'correct': correct, 'wrong': wrong, 'erasures': erased}


def archive_words(artifact, pn, generator, receipts):
    first, second = archive_first_and_prior(artifact('v076-code5-byte-words', 'E078', 'archived_code5_words.npz'),
        artifact('v078-archive-code5-links', 'E080', 'recovered_code5_words.npz'), pn, generator)
    second_ids = sorted(second)
    if second_ids != receipts['second_word_from_e078']:
        raise ValueError('Initial second-word archive differs from E091')
    received = artifact('v089-code5-directory-final', 'E091', 'code5_directory_following_words.npz')
    with np.load(received, allow_pickle=False) as source:
        for frame, packed, mask in zip(source['frame_index'], source['bits'], source['observed'], strict=True):
            bits = np.unpackbits(packed)
            if frame in second or not np.array_equal((bits[:1024] @ generator) & 1, bits[1024:]):
                raise ValueError(f'Second word fails its frozen Code 5 check in Qin frame {frame}')
            second[int(frame)] = (descramble(bits, pn), np.unpackbits(mask)[:1024].astype(bool))
    if len(second) - len(second_ids) != receipts['new_second_word_count'] or set(first) != set(second):
        raise ValueError('Qin first and second words do not have the frozen membership')
    return {frame: {0: first[frame], 1: second[frame]} for frame in sorted(first)}


def pack_observations(words, rows, directory, coordinates, pn, ratios):
    frames = [r['frame'] for r in rows]
    if sorted(words) != frames:
        raise ValueError('Code 5 memberships do not match the directory receipt')
    masks, transmitted, metadata = [], [], []
    pos = np.array([i for _, i in coordinates])
    codeword = np.array([k for k, _ in coordinates])
    for row in rows:
        pair_words = words[row['frame']]
        first, second = pair_words[0], pair_words[1]
        data, observed = first
        length = data[0]
        span = int.from_bytes(data[5:7], 'little') >> 4
        affine = directory['affine_convention']
        crc = crc_bytes(data[:length - 3], directory['crc_polynomial_including_top_bit'],
                        affine['seed'], affine['residue'])
        if (not row['crc_pass'] or data[length - 3:length] != crc or crc.hex() != row['expected_crc'] or
                (length, span) != (row['directory_length'], row['decoded_record_span']) or
                int(observed[:length * 8].sum()) != row['observed_directory_bits'] or
                not observed[0:8].all() or not observed[40:56].all()):
            raise ValueError(f'Code 5 layout or CRC receipt changes in frame {row["frame"]}')
        pair = np.stack([np.unpackbits(np.frombuffer(w, np.uint8), bitorder='little') for w in (data, second[0])])
        transmitted.append(template_signs(pair[codeword, pos], pn[pos], ratios))
        masks.append(np.stack((observed, second[1]))[codeword, pos])
        metadata.append({'frame': row['frame'], 'directory_length': length, 'record_span': span,
                         'group': row.get('group'), 'calibration': row.get('calibration')})
    return np.stack(transmitted), np.stack(masks), metadata


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    preceding = read(artifact('v133-leo-affine-transfer-final', 'E135', 'leo_affine_transfer.json'))
    relations = [r for r in preceding['relations'] if r['name'] in CANDIDATES]
    if tuple(r['name'] for r in relations) != CANDIDATES:
        raise ValueError('E135 candidate membership changed')
    signed = {tuple(c): s for c, s in zip(preceding['coordinates'], preceding['coordinate_phase_signs'], strict=True)}
    mapping = {}
    for relation in relations:
        for coord, role in zip(relation['left'] + relation['right'], relation['roles'], strict=True):
            role_id = (role['word'], role['offset'])
            if role['part'] != 'Code 5 systematic' or mapping.setdefault(tuple(coord), role_id) != role_id:
                raise ValueError('E135 candidate coordinates changed')
    coords = sorted(mapping)
    ratios = np.array([signed[c] for c in coords])
    coordinate_ids = {coord: i for i, coord in enumerate(coords)}

    body_run = 'v010-code5-parity-02'
    raw_480 = artifact(body_run, 'E012', 'code5_observations.npz')
    with np.load(raw_480, allow_pickle=False) as data:
        generator = data['parity_generator']
    model = read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    directory = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    qin = archive_words(artifact, pn, generator, directory)
    raw_250_run = 'v022-ut-replay'
    raw_250 = artifact(raw_250_run, 'E014', 'ut_observations.npz')
    words_480 = raw_record_set(raw_480, read(artifact(body_run, 'E012', 'code5_parity.json'))['words'],
                               'hard_bits', pn, generator)
    words_250 = raw_record_set(raw_250, read(artifact(raw_250_run, 'E014', 'ut_transfer.json'))['words'],
                               'hard_code5_words', pn, generator)
    sources = {'Qin': (qin, directory['archive']), '480MSps': (words_480, directory['raw']['480MSps']),
               '250MSps': (words_250, directory['raw']['250MSps'])}
    datasets = {name: pack_observations(words, rows, directory, [mapping[c] for c in coords], pn, ratios)
                for name, (words, rows) in sources.items()}
    e135_later = set(preceding['frame_groups']['Qin_after_77'])
    qin_rows = datasets['Qin'][2]
    # Anchor the archive's original real-axis sign to the known H1 prefix.
    # This comparison uses no coordinates from any LEO relation.
    prefix = word_bits(0x3048)[16:]
    pilot = 1 - 2 * encode_block(np.r_[prefix, np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.array([pn_reference(int(k), np.array([2]))[0] for k in NONPILOT[:48]]).conj()
    prefix_path = artifact('v034-qin-header-survey', 'E036', 'qin_header_symbols.npz')
    with np.load(prefix_path, allow_pickle=False) as source:
        if not np.array_equal(source['frame_index'], [r['frame'] for r in qin_rows]):
            raise ValueError('Qin prefix and Code 5 archives differ in membership')
        prefix_gain = (source['h1_symbols'][:, :48] * rotation * pilot).mean(axis=1)
    if np.any(abs(prefix_gain.real) <= 1e-10):
        raise ValueError('Qin H1 prefix cannot determine the real-axis polarity')
    inverted = {r['frame'] for r, gain in zip(qin_rows, prefix_gain, strict=True) if gain.real < 0}
    beyond = np.array([r['frame'] >= 78 and r['frame'] not in e135_later for r in qin_rows])
    calibration = beyond & np.array([r['calibration'] for r in qin_rows])
    held = beyond & np.array([not r['calibration'] for r in qin_rows])
    layouts = [(r['directory_length'], r['record_span']) for r in qin_rows]
    length24 = np.array([length == 24 for length, _ in layouts])
    selected = {'Qin_remaining': beyond, 'Qin_layout_holdout': held, 'Qin_length24_holdout': held & length24}
    replay_groups = (('Qin', 'Qin_LEO_0_77'), ('Qin', 'Qin_after_77'), ('480MSps', 'independent_480MSps'))
    results = []
    for relation in relations:
        left = tuple(coordinate_ids[tuple(c)] for c in relation['left'])
        right = tuple(coordinate_ids[tuple(c)] for c in relation['right'])
        traces = {name: relation_traces(bits, mask, left, right, relation['constant'])
                  for name, (bits, mask, _) in datasets.items()}
        qin_trace = traces['Qin']
        for name, group in replay_groups:
            rows = datasets[name][2]
            index = {r['frame']: i for i, r in enumerate(rows)}
            prior = relation['results'][group]
            for key in ('left', 'adjusted_right', 'joint'):
                frames = preceding['frame_groups'][group]
                replay = ''.join(traces[name][key][index[frame]] for frame in frames)
                size = {'left': len(left), 'adjusted_right': len(right), 'joint': len(left) + len(right)}[key]
                if name == 'Qin' and size % 2:
                    replay = ''.join(str(1 - int(bit)) if bit != '?' and frame in inverted else bit
                                     for frame, bit in zip(frames, replay, strict=True))
                if replay != prior[key]:
                    difference = [frame for frame, a, b in zip(frames, replay, prior[key], strict=True) if a != b]
                    raise ValueError(f'{relation["name"]} {key} does not replay {group}: frames {difference}')

        used = [coords[i] for i in left + right]
        bins = np.array([k if k < 512 else k - 1024 for _, k in used])
        choices = {}
        for subset, keep in selected.items():
            trace = {key: ''.join(np.array(list(value))[keep]) for key, value in qin_trace.items()}
            choices[subset] = counts(trace)
        layout_comparisons = {}
        for side in ('left', 'adjusted_right'):
            y = qin_trace[side]
            length_only = predict_by_layout([length for length, _ in layouts], y, calibration)
            length_span = predict_by_layout(layouts, y, calibration)
            layout_comparisons[side] = {subset: {'directory_only': score(length_only, y, keep),
                                                'directory_and_span': score(length_span, y, keep)}
                                        for subset, keep in selected.items() if subset != 'Qin_remaining'}
        by_layout = defaultdict(Counter)
        for layout, a, b, keep in zip(layouts, qin_trace['left'], qin_trace['adjusted_right'], beyond, strict=True):
            if keep:
                by_layout[layout][a + b] += 1
        flags = [word * 1024 + offset for word, offset in (mapping[c] for c in used)]
        result = {'name': relation['name'], 'coords': used, 'descrambled_bit_offsets': flags,
                  'center_to_center_span_hz': float(np.ptp(bins) * 234375), 'qin_sections': choices,
                  'qin_by_layout': [{'length': length, 'span': span, 'pairs': dict(histogram),
                                     'parts': [layout_role(int(bit), length, span) for bit in flags]}
                                    for (length, span), histogram in sorted(by_layout.items())],
                  'predictions': layout_comparisons,
                  'captures': {name: {**trace, **counts(trace)} for name, trace in traces.items()}}
        print(relation['name'], choices, layout_comparisons['adjusted_right']['Qin_length24_holdout'])
        results.append(result)

    result = {'schema': 'starlink-leo-layout-relation/v1',
              'input_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              'frames': {name: rows for name, (_, _, rows) in datasets.items()},
              'qin_inverted_relative_to_known_h1_prefix': sorted(inverted),
              'qin_layout_training_frames': [r['frame'] for r, keep in zip(qin_rows, calibration, strict=True) if keep],
              'qin_section_frames': {name: [r['frame'] for r, keep in zip(qin_rows, selection, strict=True) if keep]
                                     for name, selection in selected.items()},
              'relations': results}
    destination = experiment_output('E136')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'leo_layout_relation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
