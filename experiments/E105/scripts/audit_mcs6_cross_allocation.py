"""Check downstream MCS allocations predicted by the earlier MCS 6 directory."""

from collections import Counter
from functools import lru_cache
from itertools import accumulate
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_gain_transfer import decisions, four_corner_slots
from audit_mcs6_body_spans import RECORD_TAGS, span_from_entry
from audit_mcs6_directory_crc import LOW_MCS, directory_candidates, parse_compact_entries
from audit_pn_padding_starts import CacheOnly, prefix_profile
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mcs6_folded import prepare
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import as_complex
from validate_code21 import K as MCS21_SYSTEMATIC_BITS


SEED = 20261122105
SHUFFLES = 199


def ordered_boundaries(lengths, capacities):
    """Keep all forward ranges, plus the subset that can consume the list exactly."""
    cumulative = list(accumulate(lengths, initial=0))
    prefix, active = [], {0}
    for words, bytes_per_word in capacities:
        lower = (words - 1) * bytes_per_word + int(words > 1)
        upper = words * bytes_per_word
        ranges = [(i, j, cumulative[j] - cumulative[i]) for i in sorted(active)
                  for j in range(i, len(cumulative)) if lower <= cumulative[j] - cumulative[i] <= upper]
        prefix.append(ranges)
        active = {j for _, j, _ in ranges}
    complete, eligible = [], {len(lengths)}
    for ranges in reversed(prefix):
        compatible = [(i, j, size) for i, j, size in ranges if j in eligible]
        complete.append(compatible)
        eligible = {i for i, _, _ in compatible}
    return prefix, complete[::-1]


def describe_frames(artifact, early, outer, pn):
    low = read(artifact(LOW_MCS, 'E036', 'qin_low_mcs_audit.json'))['formats']['6']['words']
    transfer = read(artifact('v036-qin-mcs6-transfer', 'E038', 'qin_mcs6_transfer.json'))['words']
    directory = read(artifact('v101-mcs6-directory', 'E103', 'mcs6_directory_crc.json'))
    samples = artifact('v034-qin-low-mcs-extract', 'E036', 'qin_low_mcs_symbols.npz')
    with np.load(samples, allow_pickle=False) as archive:
        hard, erased = prepare(archive['mcs6_symbols'], low, transfer, pn[:16384])['edge']
    frames = []
    for index, evidence in enumerate(directory['mcs6']):
        if evidence['word_in_frame'] != 0:
            continue
        frame = evidence['frame']
        old = early[frame]
        candidates = directory_candidates(hard[index] ^ pn[:16384], erased[index], directory['frozen_crc'])
        if len(candidates) != 1:
            raise ValueError(f'Unable to confirm frozen directory in frame {frame}')
        header = candidates[0]
        partition = parse_compact_entries(header)
        if partition != evidence['partition'] or header[-3:].hex() != evidence['crc_hex']:
            raise ValueError(f'Directory differs from E103 in frame {frame}')
        offsets = [p for p, tag in zip(partition['entry_offsets'], partition['entry_tags'], strict=True) if tag == 3]
        if offsets[:len(old['records'])] != [r['descriptor_offset'] for r in old['records']]:
            raise ValueError(f'Assigned prefix differs from E104 in frame {frame}')
        lengths = [span_from_entry(header[p:p + 5]) for p in offsets[len(old['records']):]]
        if len(lengths) != old['remaining_unassigned_5_byte_descriptors']:
            raise ValueError(f'Remaining descriptors differ from E104 in frame {frame}')
        h2 = outer[frame]['h2']
        alloc = h2['layout20']['entries']
        if not h2['crc_ok'] or alloc[0]['mcs_candidate'] != 6 or alloc[0]['quantity'] != old['words_from_h2']:
            raise ValueError(f'Expected verified MCS 6 as first H2 allocation in frame {frame}')
        frames.append({'frame': frame, 'group': old['group'], 'calibration': old['calibration'],
                       'remaining_lengths': lengths, 'h2_entries': alloc[1:]})
    return frames


def record_header_checks(lengths, ranges, body, observed):
    hypotheses = set()
    for i, j, _ in ranges:
        offset = 0
        for size in lengths[i:j]:
            hypotheses.add((offset, size))
            offset += size
    tests = []
    for delta in (0, -1, 1, -2, 2):
        admissible, matched = [], []
        for offset, size in sorted(hypotheses):
            position = offset + delta
            if position < 0 or position + 4 > len(body) or not observed[position:position + 4].all():
                continue
            field = int.from_bytes(body[position + 2:position + 4], 'little')
            if body[position] not in RECORD_TAGS or field & 31 != 16:
                continue
            received = 4 + (field >> 5)
            admissible.append([offset, size, received])
            if received == size:
                matched.append([offset, size])
        tests.append({'shift_bytes': delta, 'record_candidates': len(hypotheses),
                      'observed_length_fields': admissible, 'matches': matched})
    return tests


def inspect_allocations(frames, geometry, gain, capacities, models, pn):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    reader = CacheOnly(ROOT / '.cache/qin-mat-pages', False)
    results = []
    with reader, h5py.File(reader, 'r') as archive:
        source = archive['yDataDec']

        @lru_cache(maxsize=1)
        def block(first):
            stop = min(first + source.chunks[0], len(source))
            return as_complex(source[first:stop, 1:, :])[..., NONPILOT % 1024]

        for row in frames:
            frame = row['frame']
            allocation = [(r['mcs_candidate'], r['quantity']) for r in row['h2_entries']]
            sizes = [(q, capacities[m] // 8) for m, q in allocation]
            lengths = row['remaining_lengths']
            prefix, complete = ordered_boundaries(lengths, sizes)
            first = frame // source.chunks[0] * source.chunks[0]
            flat = remove_header_gain(block(first)[frame - first] * rotation, signs)
            axis = complex(*gain['frames'][frame]['axis'])
            corners = four_corner_slots(flat, axis)
            layouts = geometry['word_windows_and_onsets'][frame]['allocations']
            evidence = gain['frames'][frame]['allocations']
            references = {layout['label']: (layout, proof) for layout, proof in zip(layouts, evidence, strict=True)}
            if len(references) != len(layouts):
                raise ValueError(f'Multiple archived allocations with the same MCS in frame {frame}')
            rows = []
            for index, (mcs, words) in enumerate(allocation):
                entry = {'mcs': mcs, 'words': words, 'systematic_bytes_per_word': capacities[mcs] // 8,
                         'prefix_legal_ranges': prefix[index], 'complete_legal_ranges': complete[index]}
                rows.append(entry)
                if mcs not in references:
                    continue
                layout, proof = references[mcs]
                if words != layout['count'] or proof['start'] != layout['start'] or proof['label'] != mcs:
                    raise ValueError(f'E070/E081 geometry changed in frame {frame}, MCS {mcs}')
                width = capacities[mcs]
                bits, seen, _, _ = decisions(flat, axis, layout, models, width, corners)
                reference_onset = prefix_profile(bits[-1], seen[-1], pn, width)
                raw = np.packbits((bits[:, :width] ^ pn[:width]).ravel(), bitorder='little').tobytes()
                visible = seen[:, :width].reshape(-1, 8).all(axis=-1)
                entry.update(pn_admitted=bool(words - 1 in proof['masked_transfer_stop_words']),
                             onset_bits=reference_onset['ml_start'] + (words - 1) * width,
                             header_tests=record_header_checks(lengths, prefix[index], raw, visible))
            results.append({**row, 'allocation_candidates': rows})
    pages = [reader.receipts[k] for k in sorted(reader.receipts)]
    pinned = {tuple(r['bytes']): r['sha256'] for r in gain['source']['pages']}
    if any(pinned.get(tuple(page['bytes'])) != page['sha256'] for page in pages):
        raise ValueError('Qin archive bytes differ from E081')
    return results, pages


def summarize(rows, proposed):
    totals = Counter()
    for frame, layouts in zip(rows, proposed, strict=True):
        for row, ranges in zip(frame['allocation_candidates'], layouts, strict=True):
            if not row.get('pn_admitted'):
                continue
            endpoints = {n for _, _, n in ranges}
            onset = row['onset_bits']
            exact = round(onset / 8) in endpoints
            near = any(abs(n * 8 - onset) <= 8 for n in endpoints)
            totals.update(qualified=1, feasible=bool(endpoints), exact_rounded=exact, within_one_byte=near,
                          unique=len(endpoints) == 1, unique_within_one_byte=len(endpoints) == 1 and near)
    return dict(totals)


def controls(frames):
    rng = np.random.default_rng(SEED)

    def permuted(reorder):
        options = []
        for frame in frames:
            dims = [(r['words'], r['systematic_bytes_per_word']) for r in frame['allocation_candidates']]
            ranges, _ = ordered_boundaries(reorder(frame['remaining_lengths']), dims)
            options.append(ranges)
        return summarize(frames, options)

    return {'compare_reversed': permuted(lambda a: a[::-1]),
            'compare_permuted': [permuted(lambda a: rng.permutation(a).tolist()) for _ in range(SHUFFLES)]}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    prior = read(artifact('v102-mcs6-spans', 'E104', 'mcs6_body_spans.json'))['results']
    outer = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    geometry = read(artifact('v068-pn-padding-starts', 'E070', 'pn_padding_starts.json'))
    gain = read(artifact('v079-code5-gain-masked', 'E081', 'code5_gain_transfer.json'))
    cross = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    models = constellation_models(cross, inputs)
    pn = equality_predictions(np.arange(16385), pn_bits(), cross['frozen_pn_model']).astype(bool)
    frames = describe_frames(artifact, {r['frame']: r for r in prior}, outer, pn)
    capacities = {int(k): n for k, n in geometry['marker_systematic_ends'].items()}
    capacities[21] = MCS21_SYSTEMATIC_BITS
    results, pages = inspect_allocations(frames, geometry, gain, capacities, models, pn)
    prefix = [[r['prefix_legal_ranges'] for r in f['allocation_candidates']] for f in results]
    complete = [[r['complete_legal_ranges'] for r in f['allocation_candidates']] for f in results]
    observed = {'forward': summarize(results, prefix), 'complete': summarize(results, complete),
                'seed': SEED, 'shuffle_count': SHUFFLES, **controls(results)}
    report = {'schema': 'starlink-mcs6-cross-allocation/v1', 'archive_pages': pages,
              'frames': results, 'endpoint_comparison': observed,
              'inputs_sha256': {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()}}
    output = experiment_output('E105')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mcs6_cross_allocation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Prefix-compatible endpoints:', observed['forward'])
    print('Complete allocation endpoints:', observed['complete'])
    print('Reverse descriptor order:', observed['compare_reversed'])
    print('Max shuffled within one byte:', max(r['within_one_byte'] for r in observed['compare_permuted']))


if __name__ == '__main__':
    main()
