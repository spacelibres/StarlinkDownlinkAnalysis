import unittest

import numpy as np

import project_paths
from audit_cross_mcs_record_fields import counter_pairs, inner_candidates, outer_record
from audit_cross_mcs_integrity_fields import crc32c, footer_probes


class CrossMcsRecordFieldTests(unittest.TestCase):
    def test_inner_anchor_keeps_an_erased_byte_out_of_the_fit(self):
        payload = np.frombuffer(bytes.fromhex('1020e8c0e31940ff3dd20e53006b5e3303b3ffd0'), dtype=np.uint8)
        seen = np.ones(len(payload), bool)
        self.assertEqual(inner_candidates(payload, seen), [0])
        seen[12] = False
        self.assertEqual(inner_candidates(payload, seen), [])

    def test_counter_identity_and_big_endian_carry(self):
        a = bytes.fromhex('1020e8c0e31940ff3dd20e53006b5e3303b3ffd0')
        b = a[:6] + bytes.fromhex('4100') + a[8:]
        values = counter_pairs([(4, a), (1524, b)])
        self.assertEqual(values[0]['big_endian_step'], 1)
        self.assertEqual(values[0]['little_endian_step'], 257)

    def test_outer_length_checks_encoded_flag_and_buffer_capacity(self):
        packet = bytes.fromhex('90411008') + bytes(64)
        self.assertEqual(outer_record(packet, 0), 68)
        self.assertIsNone(outer_record(packet[:-1], 0))
        self.assertIsNone(outer_record(packet[:2] + bytes.fromhex('1108') + packet[4:], 0))

    def test_public_crc_vector_and_footer_detection(self):
        body = b'123456789'
        self.assertEqual(crc32c(body), 0xe3069283)
        packet = bytes(4) + body + crc32c(body).to_bytes(4, 'little')
        self.assertTrue(any(r['name'] == 'crc32c' and r['start'] == 4 and r['byte_order'] == 'little'
                            and r['residual'] == '00000000' for r in footer_probes(packet)))
