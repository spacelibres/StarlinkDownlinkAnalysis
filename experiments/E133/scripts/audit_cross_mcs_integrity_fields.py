"""Check bounded record ends against published CRCs and the known local directory CRC."""

from binascii import crc_hqx, crc32
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_directory_checks import crc_bytes
from audit_cross_mcs_record_fields import make_pn, raw_streams
from audit_ut_repeated_data import checked_artifact, read


def crc32c(body):
    value = 0xffffffff
    for byte in body:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (0x82f63b78 if value & 1 else 0)
    return value ^ 0xffffffff


def footer_probes(record, crc24=None):
    methods = [('crc16_ibm3740', 2, lambda x: crc_hqx(x, 0xffff)), ('crc16_xmodem', 2, lambda x: crc_hqx(x, 0)),
               ('crc32_iso_hdlc', 4, crc32), ('crc32c', 4, crc32c)]
    if crc24 is not None:
        methods.append(('local_directory_crc24', 3, lambda x: int.from_bytes(crc_bytes(x, *crc24), 'big')))
    output = []
    for name, width, algorithm in methods:
        for start in (0, 4):
            body, trailer = record[start:-width], record[-width:]
            predicted = algorithm(body)
            for order in ('little', 'big'):
                residual = predicted ^ int.from_bytes(trailer, order)
                output.append({'name': name, 'start': start, 'byte_order': order,
                               'body_sha256': sha256(body).hexdigest(), 'residual': f'{residual:0{2 * width}x}'})
    return output


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    lengths = read(artifact('v131-cross-mcs-records', 'E133', 'cross_mcs_record_fields.json'))['raw']
    labels = read(artifact('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json'))
    directory = read(artifact('v089-code5-directory-final', 'E091', 'code5_directory_checks.json'))
    affine = directory['affine_convention']
    crc24 = directory['crc_polynomial_including_top_bit'], affine['seed'], affine['residue']
    records, buckets = [], defaultdict(list)
    for code, frame, resource, reserved, data in raw_streams(artifact, labels['frame_split']['reserved'],
                                                          make_pn(labels['frozen_pn_model'])):
        source = next(r for r in lengths[str(code)] if r['frame'] == frame and r['resource'] == resource)
        for interval in source['outer_chain']:
            a, b = interval['start'], interval['stop']
            raw = data[a:b].tobytes()
            if not raw.startswith(bytes.fromhex(interval['outer_prefix_hex'])):
                raise ValueError('Source length receipt no longer describes the raw start')
            tests = footer_probes(raw, crc24)
            idx = len(records)
            records.append({'mcs': code, 'frame': frame, 'receiver_reserved': bool(reserved),
                            'interval': [a, b], 'probes': tests})
            for r in tests:
                key = (b - a, r['name'], r['start'], r['byte_order'], r['residual'])
                buckets[key].append((idx, r['body_sha256']))
    repeated = [{'scope': key, 'records': [j for j, _ in rows]}
                for key, rows in sorted(buckets.items()) if len({h for _, h in rows}) > 1]
    result = {'schema': 'starlink-outer-record-integrity/v1', 'records': records, 'same_length_distinct_body_residual_matches': repeated,
              'direct_matches': sum(int(p['residual'], 16) == 0 for record in records for p in record['probes']),
              'tests': sum(len(r['probes']) for r in records),
              'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    target = experiment_output('E133')
    target.mkdir(parents=True, exist_ok=True)
    (target / 'cross_mcs_integrity_fields.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(len(records), 'records,', result['tests'], 'probes, direct matches', result['direct_matches'],
          'repeated distinct-body residues', len(repeated), flush=True)


if __name__ == '__main__':
    main()
