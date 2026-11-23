"""Recheck published headers, directory CRCs and shortened high parity."""

import argparse
from array import array
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VECTOR_PATH = HERE / 'data' / 'verification_vectors.json'
ENDS = {71: 10752, 75: 11008, 78: 11520, 81: 11776}
REFLECT = bytes(int(f'{i:08b}'[::-1], 2) for i in range(256))


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def crc8(data):
    value = 0xff
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = ((value << 1) & 0xff) ^ (0x4d if value & 0x80 else 0)
    return value


def crc24(prefix, poly, seed, residue):
    augmented = int.from_bytes(prefix.translate(REFLECT), 'big') << 24
    value = augmented ^ (seed << (8 * (len(prefix) + 3)))
    while value.bit_length() >= poly.bit_length():
        value ^= poly << (value.bit_length() - poly.bit_length())
    return (value ^ residue).to_bytes(3, 'big').translate(REFLECT)


def unpack(hexadecimal):
    return [int(bit) for value in bytes.fromhex(hexadecimal) for bit in f'{value:08b}']


def export():
    import numpy as np

    hashes = {}

    def evidence(run, experiment, file):
        path = ROOT / 'runs' / run / 'artifacts' / 'experiments' / experiment / 'results' / file
        hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        return path

    headers = []
    for capture in ('ut250', 'legacy480', 'qin'):
        source = load(evidence('v181-code5-json-v2', 'E174', f'{capture}_control_records.v2.json'))
        for row in (r for r in source['frames'] if r['header_1']['crc8_pass']
                    and r['header_2']['crc8_pass']):
            headers.append({
                'capture': capture, 'frame': row['source_frame_index'],
                'h1_hex': ''.join(w[2:] for w in row['header_1']['words_hex']),
                'h2_hex': ''.join(w[2:] for w in row['header_2']['words_hex']),
                'h1_allocations': row['header_1']['allocations'],
                'h2_resource_start': row['header_2_resource_start'],
            })
            break
    config = load(evidence('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    affine = config['affine_convention']
    crc = [config['crc_polynomial_including_top_bit'], affine['seed'], affine['residue']]
    rows = load(evidence('v252-qin-mcs57-code5', 'E245', 'qin_mcs57_code5.json'))['allocations']
    directories = [
        {**{k: r[k] for k in ('frame', 'group', 'resource')},
         'bytes_hex': r['finite_prefix']['accepted_bytes_hex']}
        for r in rows if r['finite_prefix'].get('historical_status') == 'new_coordinate'
    ]
    run = 'v254-qin-partial-core-transfer'
    rows = load(evidence(run, 'E247', 'qin_partial_core_transfer.json'))['words']
    with np.load(evidence('v126-mcs66-full-code', 'E128', 'mcs66_full_checks.npz'),
                 allow_pickle=False) as matrix:
        supports = matrix['row_support'].astype('<u2').tobytes().hex()
    groups = {mcs: set() for mcs in ENDS}
    cores = []
    with np.load(evidence(run, 'E247', 'qin_partial_core_transfer_observations.npz'),
                 allow_pickle=False) as source:
        for row in rows:
            mcs, group = row['mcs'], row['group']
            if 'new_array_index' not in row or group in groups[mcs] or len(groups[mcs]) == 2:
                continue
            groups[mcs].add(group)
            index = row['new_array_index']
            stored = source['bits'][index]
            seen = source['seen'][index]
            end = ENDS[mcs] // 8
            assert np.all(seen[:end] == 255) and np.all(seen[1536:] == 255)
            cores.append({
                **{k: row[k] for k in ('mcs', 'group', 'frame', 'resource', 'word')},
                'systematic_scrambled_hex': stored[:end].tobytes().hex(),
                'verified_high_parity_hex': stored[1536:].tobytes().hex(),
                'direct_low_bits': sum(int(i).bit_count() for i in seen[end:1536]),
                'waveform_receipts': {arm: result['waveform'] for arm, result in row['methods'].items()
                                      if result['status'] == 'verified_core'},
            })
    document = {
        'schema': 'starlink-stage-ii-verification-vectors/v1',
        'scope': 'Extracted measured-stage values. Raw IQ and unknown low parity are omitted.',
        'sources_sha256': hashes, 'headers': headers,
        'directory_crc24_parameters': dict(zip(('poly', 'seed', 'residue'), crc, strict=True)),
        'mcs57_crc24_directories': directories,
        'mother_check_support_le_u16_hex': supports, 'partial_high_cores': cores,
    }
    VECTOR_PATH.write_bytes((json.dumps(document, indent=2, allow_nan=False) + '\n').encode())


def verify():
    data = load(VECTOR_PATH)
    assert {r['capture'] for r in data['headers']} == {'ut250', 'legacy480', 'qin'}
    for row in data['headers']:
        for key in ('h1_hex', 'h2_hex'):
            raw = bytes.fromhex(row[key])
            assert crc8(raw[:-1]) == raw[-1], row
            assert crc8(bytes([raw[0] ^ 1]) + raw[1:-1]) != raw[-1], row
        assert all(a['mcs'] == 5 for a in row['h1_allocations']), row
        count = sum(a['codeword_count'] for a in row['h1_allocations'])
        assert 228 + 2048 * count == row['h2_resource_start'], row
    directories = data['mcs57_crc24_directories']
    assert len({(r['frame'], r['resource']) for r in directories}) == 8
    for row in directories:
        raw = bytes.fromhex(row['bytes_hex'])
        assert raw[0] == len(raw), row
        params = data['directory_crc24_parameters']
        assert crc24(raw[:-3], **params) == raw[-3:], row
        assert crc24(bytes([raw[0] ^ 1]) + raw[1:-3], **params) != raw[-3:], row
    indices = array('H')
    indices.frombytes(bytes.fromhex(data['mother_check_support_le_u16_hex']))
    if sys.byteorder == 'big':
        indices.byteswap()
    assert len(indices) == 4096 * 14 and max(indices) < 16384
    checks = [indices[i:i + 14] for i in range(0, len(indices), 14)]
    cores = data['partial_high_cores']
    assert Counter(r['mcs'] for r in cores) == {mcs: 2 for mcs in ENDS}
    for row in cores:
        system = unpack(row['systematic_scrambled_hex'])
        high = unpack(row['verified_high_parity_hex'])
        assert len(system) == ENDS[row['mcs']] and len(high) == 4096
        word = [0] * (12288 - len(system)) + system + high
        assert all(not (sum(word[j] for j in equation) & 1) for equation in checks), row['resource']
        shifted = word[:12288] + high[-1:] + high[:-1]
        assert any(sum(shifted[j] for j in equation) & 1 for equation in checks), row['resource']
    print('Verified 6 CRC-8 headers, 3 H2 offsets, 8 CRC-24 directories and 8 × 4096'
          ' shortened parity equations, with altered controls rejected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-local', action='store_true', help='Requires the full local archive')
    if parser.parse_args().export_local:
        export()
    verify()
