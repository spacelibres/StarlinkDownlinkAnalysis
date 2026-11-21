"""Check Code 5 record lengths against the outer allocation and raw padding."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_archive_code5_recovery import normalize_axis
from audit_code5_byte_records import BIT_LENGTH, START_RESOURCE, bpsk_mask, erasure_word
from audit_code5_directory_checks import (SECOND_MARKER, archive_first_and_prior, crc_bytes,
                                          recover_partial_parity)
from audit_code5_frame_count import descramble
from audit_code5_to_code21_edge import carrier_axis
from audit_h2_projection import GRID, visible_positions
from audit_leo_code5_projection import field_rank, nominal_capture_windows, selected_windows
from audit_leo_pilot_archive import sha256
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


WORD_BYTES = 128
TAG_A = bytes.fromhex('01 f0 0f')
TAG_B = bytes.fromhex('10 00 ff 60')


@dataclass(frozen=True)
class Directory:
    length: int
    record_lengths: tuple[int, ...]
    padding_bytes: int

    @property
    def record_starts(self):
        offset = self.length
        starts = []
        for size in self.record_lengths:
            starts.append(offset)
            offset += size
        return tuple(starts)

    @property
    def total_bytes(self):
        return self.length + sum(self.record_lengths) + self.padding_bytes


def parse_directory(block):
    if len(block) < 2 or block[1] != 0 or not 18 <= block[0] <= min(len(block), WORD_BYTES):
        raise ValueError('Invalid Code 5 directory prefix or length')
    length = block[0]
    body = block[2:length - 3]
    entries = []
    while True:
        tag = TAG_A if len(entries) % 2 == 0 else TAG_B
        if not body.startswith(tag):
            break
        encoded = body[len(tag):len(tag) + 2]
        if len(encoded) != 2:
            raise ValueError('Truncated record length')
        value = int.from_bytes(encoded, 'little')
        if tag == TAG_A:
            if value & 15 != 6:
                raise ValueError('Unexpected length flags after 01 f0 0f')
            value >>= 4
        if not value:
            raise ValueError('Zero-length record')
        entries.append(value)
        body = body[len(tag) + 2:]
    if len(entries) < 2:
        raise ValueError('Missing first record pair')
    if len(entries) % 2:
        if len(body) < 3 or body[0] != 0x80 or body[2] >> 4 != 0xF:
            raise ValueError('Invalid three-byte padding field')
        padding = int.from_bytes(body[1:3], 'little') & 0xFFF
        filler = body[3:]
    else:
        if len(body) < 2 or body[0] & 15 != 8:
            raise ValueError('Invalid two-byte padding field')
        padding = int.from_bytes(body[:2], 'little') >> 4
        filler = body[2:]
    if any(value != 0xFF for value in filler):
        raise ValueError('Unexpected bytes before the directory CRC')
    return Directory(length, tuple(entries), padding)


def quantity(header, layout_key='layout20'):
    entry = header[layout_key]['entries'][0]
    if not header['crc_ok'] or entry['mcs_candidate'] != 5:
        raise ValueError('Expected CRC-valid H1 and first allocation of Code 5')
    return entry['quantity']


def raw_words(path, metadata, key, counts, pn, generator):
    frames = defaultdict(dict)
    with np.load(path, allow_pickle=False) as archive:
        for row, bits in zip(metadata, archive[key], strict=True):
            frame = row.get('frame', row.get('candidate_index'))
            word = row['word']
            if word >= counts[frame] or row['start_resource'] != START_RESOURCE + word * BIT_LENGTH:
                continue
            if word in frames[frame] or np.any(((bits[:1024] @ generator) & 1) != bits[1024:]):
                raise ValueError(f'Duplicate or invalid raw Code 5 word: {frame}/{word}')
            frames[frame][word] = descramble(bits, pn), np.ones(1024, bool)
    return frames


def recover_archive(counts, first, prior_second, pn, generator):
    max_words = max(counts.values())
    slots = (START_RESOURCE + max_words * BIT_LENGTH + len(NONPILOT) - 1) // len(NONPILOT)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, slots + 2)) for k in NONPILOT]).conj()
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    starts = [START_RESOURCE + w * BIT_LENGTH for w in range(max_words)]
    clip = [carrier_axis(start, BIT_LENGTH) == 511 for start in starts]
    reader = CacheOnly(ROOT / '.cache/qin-mat-pages', False)
    frames, coverage = {}, Counter()
    with reader, h5py.File(reader, 'r') as archive:
        source = archive['yDataDec']
        if source.shape != (len(counts), 301, 1024):
            raise ValueError('Published archive dimensions changed')
        for block in range(0, len(counts), source.chunks[0]):
            stop = min(block + source.chunks[0], len(counts))
            grid = as_complex(source[block:stop, 1:slots + 1, :])[..., NONPILOT % 1024]
            for frame, symbols in enumerate(grid, block):
                words = {0: first[frame]}
                if frame in prior_second:
                    words[1] = prior_second[frame]
                corrected = remove_header_gain(symbols * rotation, signs)
                for index in range(1, counts[frame]):
                    if index in words:
                        coverage['historical_second_word'] += 1
                        continue
                    values = normalize_axis(corrected[starts[index]:starts[index] + BIT_LENGTH])
                    if values is None:
                        coverage['unstable_axis'] += 1
                        continue
                    values[clip[index]] = 0
                    recovered = erasure_word(values, generator)
                    if recovered is None:
                        recovered = recover_partial_parity(values, generator)
                        status = 'partial_parity' if recovered is not None else 'unresolved'
                    else:
                        status = 'standard_parity'
                    coverage[status] += 1
                    if recovered is not None:
                        words[index] = descramble(recovered[0], pn), bpsk_mask(values)[:1024]
                frames[frame] = words
    return frames, dict(coverage), [reader.receipts[p] for p in sorted(reader.receipts)]


def span(words, start, length):
    """Extract a range only when every intersected word was accepted."""
    low, offset = divmod(start, WORD_BYTES)
    high = (start + length - 1) // WORD_BYTES
    if start < 0 or length <= 0 or any(i not in words for i in range(low, high + 1)):
        return None, None
    segments = [words[i] for i in range(low, high + 1)]
    data = b''.join(part for part, _ in segments)[offset:offset + length]
    mask = np.concatenate([observed for _, observed in segments])[8 * offset:8 * (offset + length)]
    return data, int(mask.sum())


def frame_record(frame, count, words, crc):
    data, first_mask = words[0]
    directory = parse_directory(data)
    length = directory.length
    affine = crc['affine_convention']
    expected = crc_bytes(data[:length - 3], crc['crc_polynomial_including_top_bit'],
                         affine['seed'], affine['residue'])
    if expected != data[length - 3:length]:
        raise ValueError(f'Unvalidated directory CRC for frame {frame}')
    padding_start = directory.total_bytes - directory.padding_bytes
    padding, observed = span(words, padding_start, directory.padding_bytes)
    if directory.padding_bytes == 0:
        padding, observed = b'', 0
    start_second = directory.record_starts[1]
    marker, marker_observed = span(words, start_second, len(SECOND_MARKER))
    third, third_observed = (None, None)
    if len(directory.record_lengths) >= 3:
        third, third_observed = span(words, directory.record_starts[2], 1)
    capacity_pass = directory.total_bytes == WORD_BYTES * count
    shifted_padding = {}
    for shift in (-2, -1):
        candidate, _ = span(words, padding_start + shift, directory.padding_bytes)
        shifted_padding[str(shift)] = None if candidate is None else not any(candidate)
    return {'frame': frame, 'word_count_h1': count, 'available_words': sorted(words),
            'complete_allocation': len(words) == count, 'directory_length': length,
            'directory_observed_bits': int(first_mask[:8 * length].sum()),
            'lengths': directory.record_lengths, 'starts': directory.record_starts,
            'predicted_bytes': directory.total_bytes, 'capacity_pass': capacity_pass,
            'padding_start': padding_start, 'padding_bytes': directory.padding_bytes,
            'padding_zero': None if padding is None else not any(padding),
            'observed_padding_bits': observed, 'early_padding_zero': shifted_padding,
            'second_marker_pass': None if marker is None else marker == SECOND_MARKER,
            'second_marker_observed_bits': marker_observed,
            'third_record_starts_with_01_or_02': None if third is None else third in (b'\x01', b'\x02'),
            'third_record_first_byte_observed_bits': third_observed}


def pilot_archive_windows(pilot_receipt, source_hashes):
    for row in pilot_receipt['datasets']:
        relative = Path('reports/2026_09_28_pilot_split_transfer') / row['dataset'] / 'spec.json'
        path = EXTERNAL_LEO_ROOT / relative
        digest = sha256(path)
        if digest != row['inputs_sha256']['spec.json']:
            raise ValueError(f'LEO pilot spec no longer matches E099: {relative}')
        source_hashes[relative.as_posix()] = digest
        spec = read(path)
        for window in spec['windows']:
            if window['edge'] != 'upper':
                raise ValueError(f'Expected upper pilot edge: {relative}')
            cfo = window['acquisition_bound_cfo_hz']
            carriers = NONPILOT[abs((NONPILOT - 491.5) * GRID + cfo) <= spec['sample_rate_hz'] / 2]
            yield {'kind': 'recorded_pilot_spec_projection', 'carriers': carriers.tolist(),
                   'labels': [f'{row["dataset"]}/{spec["session_id"]}/{window["visit_index"]}/RX{window["receiver_id"]}']}


def narrowband_windows(lengths, generator, catalogue, pilot_receipt, source_hashes):
    results = []
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    windows = (selected_windows(source_hashes) + nominal_capture_windows(catalogue)
               + list(pilot_archive_windows(pilot_receipt, source_hashes)))
    bits = np.arange(8, 1024)
    for window in windows:
        positions = visible_positions(START_RESOURCE, BIT_LENGTH, window['carriers'])
        equations = code[list(positions), 8:]
        by_length = {}
        for length in lengths:
            target = bits < length * 8
            by_length[str(length)] = {'direct_directory_bits_after_pointer': sum(8 <= i < length * 8 for i in positions),
                                      'directory_rank_after_pointer': field_rank(equations, target)}
        results.append({**window, 'by_directory_length': by_length,
                        'pointer_direct_bits': [i for i in positions if i < 8],
                        'second_length_direct_bits': [i for i in positions if 88 <= i < 104],
                        'second_length_rank': field_rank(equations, (bits >= 88) & (bits < 104))})
    return results


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model = read(artifact('v071-pn-parity-order', 'E073', 'pn_parity_order.json'))['pn_model']
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    old = 'v010-code5-parity-02'
    old_iq = artifact(old, 'E012', 'code5_observations.npz')
    with np.load(old_iq, allow_pickle=False) as archive:
        generator = archive['parity_generator']
    first, second = archive_first_and_prior(
        artifact('v076-code5-byte-words', 'E078', 'archived_code5_words.npz'),
        artifact('v078-archive-code5-links', 'E080', 'recovered_code5_words.npz'), pn, generator)
    directory_run = 'v089-code5-directory-final'
    crc = read(artifact(directory_run, 'E091', 'code5_directory_checks.json'))
    with np.load(artifact(directory_run, 'E091', 'code5_directory_following_words.npz'), allow_pickle=False) as following:
        for frame, packed, mask in zip(following['frame_index'], following['bits'], following['observed'], strict=True):
            bits = np.unpackbits(packed)
            if frame in second or np.any(((bits[:1024] @ generator) & 1) != bits[1024:]):
                raise ValueError(f'Duplicate or invalid archived second word: {frame}')
            second[int(frame)] = descramble(bits, pn), np.unpackbits(mask)[:1024].astype(bool)
    headers = read(artifact('v034-qin-header-survey', 'E036', 'qin_header_survey.json'))['frames']
    counts = {i: quantity(row['h1']) for i, row in enumerate(headers)}
    archived, coverage, pages = recover_archive(counts, first, second, pn, generator)
    prior_pages = {tuple(page['bytes']): page['sha256'] for page in crc['archive_source']['pages']}
    if any(prior_pages.get(tuple(page['bytes'])) != page['sha256'] for page in pages):
        raise ValueError('Qin cache SHA-256 differs from E091')
    split = {row['frame']: (row['group'], row['calibration']) for row in crc['archive']}
    archive_rows = [frame_record(i, count, archived[i], crc) for i, count in counts.items()]
    for row in archive_rows:
        row['group'], row['calibration'] = split[row['frame']]

    legacy_headers = read(artifact('v008-complete-header-layout-02', 'E010', 'full_header_layout.json'))['frames']
    legacy_counts = {r['legacy_frame_index']: quantity({**r['first_header'], 'layout20': r['first_layout20']})
                     for r in legacy_headers}
    ut = 'v022-ut-replay'
    ut_metadata = read(artifact(ut, 'E014', 'ut_transfer.json'))
    ut_counts = {r['candidate_index']: quantity(r['h1']) for r in ut_metadata['candidates']}
    raw = {}
    for label, path, metadata, key, allocation in (
        ('480MSps', old_iq, read(artifact(old, 'E012', 'code5_parity.json'))['words'], 'hard_bits', legacy_counts),
        ('250MSps', artifact(ut, 'E014', 'ut_observations.npz'), ut_metadata['words'], 'hard_code5_words', ut_counts),
    ):
        frames = raw_words(path, metadata, key, allocation, pn, generator)
        raw[label] = [frame_record(i, count, frames[i], crc) for i, count in sorted(allocation.items())]

    catalogue = read(artifact('v069-leo-recording-coverage', 'E071', 'leo_recording_coverage.json'))
    pilot_receipt = read(artifact('v097-leo-pilot-archive', 'E099', 'leo_pilot_archive.json'))
    external = {}
    lengths = sorted({row['directory_length'] for row in archive_rows})
    windows = narrowband_windows(lengths, generator, catalogue, pilot_receipt, external)
    record = {'schema': 'starlink-code5-directory-records/v1',
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
              'external_inputs_sha256': external,
              'archive_source': {'url': URL, 'etag': ETAG, 'bytes': SIZE, 'pages': pages},
              'archive_word_recovery': coverage, 'archive': archive_rows, 'raw': raw, 'leo_windows': windows}
    output = experiment_output('E100')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'code5_directory_records.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    for label, rows in [('Qin', archive_rows), *raw.items()]:
        for field in ('capacity_pass', 'complete_allocation', 'padding_zero', 'second_marker_pass'):
            print(label, field, dict(Counter(row[field] for row in rows)))
    print('Qin subsequent word recovery:', coverage)


if __name__ == '__main__':
    main()
