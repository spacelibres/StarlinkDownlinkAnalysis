import unittest
import zlib

import numpy as np

import project_paths
from audit_mcs85_counter import field_distribution, header_checks, score_pair, tagged_pairs


class MCS85CounterTests(unittest.TestCase):
    def test_field_projection_across_a_systematic_word_boundary(self):
        rng = np.random.default_rng(93)
        raw = rng.integers(0, 2, size=(2, 45), dtype=np.uint8)
        pn = rng.integers(0, 2, size=45, dtype=np.uint8)
        bit_number = np.arange(16) + 32
        word, pos = np.divmod(bit_number, 40)
        raw[word, pos] = np.unpackbits(np.array([0x40, 0xff], dtype=np.uint8), bitorder='little') ^ pn[pos]
        symbols = np.arange(32)[raw.reshape(2, 9, 5) @ np.array([16, 8, 4, 2, 1])].astype(complex)
        result = field_distribution(symbols, np.arange(32).astype(complex), pn, 40, 4, .001)
        self.assertEqual(int(np.argmax(result)), 0x40ff)

    def test_chain_labels_do_not_use_counter_and_chain_stops_at_missing_record(self):
        rows = [{'byte_offset': position, 'label_hex': label, 'candidate_counter_be': 0}
                for position, label in zip((0, 8, 18, 26, 80, 88, 96), ('abcd', '2468', 'abcd', 'abcd', 'abcd', 'abcd', '0000'))]
        steps = tagged_pairs(rows, {'abcd': 8, '2468': 10})
        self.assertEqual([(p['from'], p['to'], p['global_step']) for p in steps], [(0, 18, 2), (80, 88, 1)])

    def test_ethernet_fcs_calibration(self):
        payload = b'\xaa' * 12 + b'\x08\x00' + b'\x00' * 46
        frame = payload + zlib.crc32(payload).to_bytes(4, 'little')
        result = header_checks(np.frombuffer(frame, dtype=np.uint8), 0, len(frame))
        self.assertIn([0, 0], result['ethernet_fcs_hits'])
        self.assertEqual(result['ipv4_checksum_hits'], [])

    def test_ipv4_checksum_and_record_extent(self):
        header = bytes.fromhex('450000541e49400040019b0cc0a80002c0a80001')
        data = np.frombuffer(b'\x00'*12 + header + b'\x00'*64, dtype=np.uint8)
        result = header_checks(data, 0, len(data))
        self.assertEqual(result['ipv4_extent_hits'], [[12, 0]])

    def test_ipv6_requires_the_declared_extent(self):
        header = bytes.fromhex('6000000000081140') + b'\x00'*32
        packet = b'\x00'*10 + header + b'\x00'*8
        data = np.frombuffer(packet + b'\x00', dtype=np.uint8)
        self.assertEqual(header_checks(data, 0, len(packet))['ipv6_extent_hits'], [[10, 0]])
        self.assertEqual(header_checks(data, 0, len(data))['ipv6_extent_hits'], [])

    def test_expected_full_width_carry_differs_from_low_byte_wrap(self):
        a, b = np.zeros(65536), np.zeros(65536)
        a[0x40ff] = b[0x4100] = 1
        result = score_pair(a, b, ordinal=2)
        self.assertEqual(result['steps']['be']['1'], 1.)
        self.assertEqual(result['carry_score'], 1.)
        self.assertEqual(result['no_carry_model_score'], 0.)
        self.assertEqual(result['all_record_ordinal_score'], 0.)
