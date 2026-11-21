"""Check whether the frozen Code 5 directory CRC also verifies MCS 6."""

from collections import Counter
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_directory_checks import REFLECT_BYTE, crc_bytes
from audit_h2_projection import visible_positions
from audit_qam_pn15 import equality_predictions
from audit_qin_low_codes import qam_words
from audit_qin_mcs6_folded import prepare
from audit_qin_mcs6_structure import record_candidate
from audit_qin_mcs6_transfer import ANCHOR, positions_of
from audit_sparse_padding_checks import cohort
from audit_sparse_qc_orbits import predict_orbit
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


MAX_ERASURES = 5
FORMATS = (6, 35, 42, 93)
LOW_MCS = 'v034-qin-low-mcs-audit'
MCS70_SEED = (773, 1104, 1977)


def pack(bits):
    return np.packbits(bits, bitorder='little').tobytes()


def passes_crc(header, crc):
    return bool(len(header) >= 4 and crc_bytes(header[:-3], *crc) == header[-3:])


def directory_candidates(bits, erased, crc):
    prefix = np.arange(8)
    found = []
    if len(bits) < 96 or np.any(bits[8:16] & ~erased[8:16]):
        return found
    for length in range(12, min(len(bits) // 8, 252) + 1, 4):
        expected = (length >> prefix) & 1
        if np.any((bits[:8] != expected) & ~erased[:8]):
            continue
        limit = length * 8
        missing = np.flatnonzero(erased[:limit])
        if len(missing) > MAX_ERASURES:
            continue
        candidate = bits[:limit].copy()
        for values in product((0, 1), repeat=len(missing)):
            candidate[missing] = values
            header = pack(candidate)
            if header[0] == length and header[1] == 0 and passes_crc(header, crc):
                found.append(header)
    return found


def parse_compact_entries(header):
    # E037 names tags in the reversed byte representation. The same bits are 3 and 7 in CRC order.
    grammar = record_candidate(header[:-3].translate(REFLECT_BYTE))
    entries = grammar['records']
    stop = grammar['stop_byte']
    tags = [3 if item['tag'] == 0xC else 7 for item in entries]
    tail = header[stop:-3]
    return {'entry_offsets': [e['start'] for e in entries], 'entry_tags': tags,
            'n_five_byte': grammar['c_records'], 'n_two_byte': grammar['e_records'],
            'two_byte_entries_precede_five': all(tag != 7 or (i + 1 < len(tags) and tags[i + 1] == 3)
                                                for i, tag in enumerate(tags)),
            'tail_offset': stop, 'tail_and_filler_fit': len(tail) >= 2 and tail[0] & 15 == 15
                                                          and all(b == 0xFF for b in tail[2:]),
            'filler_count': max(0, len(tail) - 2)}


def audit_mcs6(meta, frames, starts, source, pn, wrong_pn, transfer, crc):
    states = prepare(source, meta, transfer, pn)
    original, _ = states['raw']
    corrected, erased = states['edge']
    no_erasures = np.zeros(original.shape[1], bool)
    seen = Counter()
    records = []
    for i, (m, frame, start) in enumerate(zip(meta, frames, starts, strict=True)):
        if m['frame'] != frame or transfer[i]['frame'] != frame:
            raise ValueError('MCS 6 frame ordering changed')
        frame = int(frame)
        seen[frame] += 1
        raw = original[i] ^ pn
        after = corrected[i] ^ pn
        direct = directory_candidates(after, no_erasures, crc)
        recovered = directory_candidates(after, erased[i], crc)
        raw_hits = directory_candidates(raw, no_erasures, crc)
        wrong = directory_candidates(corrected[i] ^ wrong_pn, erased[i], crc)
        reflected = np.unpackbits(np.packbits(after), bitorder='little')
        wrong_order = directory_candidates(reflected, no_erasures, crc)
        row = {'frame': frame, 'word_in_frame': seen[frame] - 1, 'resource_start': int(start),
               'group': m['group'], 'calibration': m['calibration'], 'raw_crc_matches': len(raw_hits),
               'edge_hard_crc_matches': len(direct), 'crc_matches_with_known_mask': len(recovered),
               'wrong_pn_crc_matches': len(wrong), 'wrong_byte_order_crc_matches': len(wrong_order)}
        if len(recovered) != 1:
            records.append(row)
            continue
        header = recovered[0]
        length = len(header)
        unknown = erased[i, :length * 8]
        observed_flips = np.flatnonzero((original[i, :length * 8] != corrected[i, :length * 8]) & ~unknown)
        plaintext = pack(after)
        motif = positions_of(plaintext[length:].translate(REFLECT_BYTE), ANCHOR)
        row.update(length=length, prefix_2_through_4=header[2:5].hex(),
                   erased_coordinates=np.flatnonzero(unknown).tolist(), all_bits_observed=not unknown.any(),
                   observed_corrected_coordinates=observed_flips.tolist(),
                   observed_corrected_edge_only=set(observed_flips) <= set(transfer[i]['carrier_transfer']['coordinates']),
                   crc_hex=header[-3:].hex(), partition=parse_compact_entries(header),
                   motif_offsets_after_directory=motif,
                   shifted_one_byte_crc=passes_crc(plaintext[1:length + 1], crc))
        records.append(row)
    return records


def mcs70_control(artifact):
    run = 'v080-gain-padded-words'
    meta = read(artifact(run, 'E082', 'gain_padded_words.json'))['cohorts']
    rows = meta['short_70']
    ordinary = meta['nonterminal_70']
    use = np.array([not r['calibration'] for r in ordinary])
    last = np.array([r['calibration'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    with np.load(artifact(run, 'E082', 'gain_padded_words.npz'), allow_pickle=False) as z:
        padded = cohort(z, 'short_70', rows)
        bits, mask = cohort(z, 'nonterminal_70', ordinary)
    return predict_orbit(((1, 0), (2, 331), (3, 180)), 512, {MCS70_SEED}, (*padded, last, groups),
                         (bits[use], mask[use], np.array([r['group'] for r in ordinary])[use]))


def leo_projection(windows, records):
    validated = [r for r in records if r.get('length')]
    result = []
    for window in windows:
        rows = []
        for r in validated:
            width = 8 * r['length']
            bits = visible_positions(r['resource_start'], width, window['carriers'])
            rows.append({'frame': r['frame'], 'pointer_bits': sum(i < 8 for i in bits), 'directory_bits': len(bits),
                         'fraction': len(bits) / width, 'full': len(bits) == width})
        result.append({'kind': window['kind'], 'labels': window['labels'], 'rows': rows})
    return result


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    audit = read(artifact(LOW_MCS, 'E036', 'qin_low_mcs_audit.json'))
    model = audit['pn_model']
    pn = equality_predictions(np.arange(16384), pn_bits(), model)
    wrong_pn = equality_predictions(np.arange(16384), pn_bits(), {**model, 'phase': model['phase'] + 1})
    crc_model = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    affine = crc_model['affine_convention']
    crc = crc_model['crc_polynomial_including_top_bit'], affine['seed'], affine['residue']
    transfer = read(artifact('v036-qin-mcs6-transfer', 'E038', 'qin_mcs6_transfer.json'))['words']
    cross = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))['cross32_mapping']
    points = np.array([complex(row['i'], row['q']) for row in cross]) / np.sqrt(20)
    labels = labels_as_bits(np.array([row['label'] for row in cross]), 5)
    source_run = 'v034-qin-low-mcs-extract'
    samples = artifact(source_run, 'E036', 'qin_low_mcs_symbols.npz')
    controls = {}
    with np.load(samples, allow_pickle=False) as z:
        records = audit_mcs6(audit['formats']['6']['words'], z['mcs6_frames'], z['mcs6_resource_start'],
                             z['mcs6_symbols'], pn, wrong_pn, transfer, crc)
        for mcs in FORMATS[1:]:
            data = qam_words(z[f'mcs{mcs}_symbols'], mcs, points, labels)
            bits = data[:, :len(pn)] ^ pn[:data.shape[1]]
            masks = np.zeros(bits.shape[1], dtype=bool)
            checks = [len(directory_candidates(row, masks, crc)) for row in bits]
            controls[mcs] = {'words': len(checks), 'matching_word_indices': [i for i, n in enumerate(checks) if n]}
    windows = read(artifact('v098-code5-records-final', 'E100', 'code5_directory_records.json'))['leo_windows']
    report = {'schema': 'starlink-mcs6-shared-directory-crc/v1', 'frozen_crc': list(crc),
              'maximum_enumerated_erasures': MAX_ERASURES, 'mcs6': records, 'other_formats_direct_only': controls,
              'mcs70_exploratory_rotation_negative': mcs70_control(artifact), 'leo_direct_projection': leo_projection(windows, records),
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    output = experiment_output('E103')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mcs6_directory_crc.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for field in ('raw_crc_matches', 'edge_hard_crc_matches', 'crc_matches_with_known_mask',
                  'wrong_pn_crc_matches', 'wrong_byte_order_crc_matches'):
        print(field, dict(Counter(row[field] for row in records)))
    print('Other format checks', controls)


if __name__ == '__main__':
    main()
