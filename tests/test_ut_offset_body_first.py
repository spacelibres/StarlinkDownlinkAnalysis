import unittest

import numpy as np

import project_paths
from audit_ut_offset_body_markers import body_paths, normalized_sliding_match
from decode_ut_offset_body_first import check_crc_tail, header_candidates, information_score
from audit_ut_offset_constrained_h2 import coded_bits
from educational_fec_demo import crc8, word_bits


class UtOffsetBodyFirstTests(unittest.TestCase):
    def test_scanner_returns_start_coordinate_and_signed_score(self):
        pattern = np.array([1, 1j, 1, -1j])
        data = np.r_[np.zeros(7), pattern, np.zeros(9), -pattern]
        scores = normalized_sliding_match(data, pattern)
        self.assertEqual(scores.argmax(), 7)
        self.assertAlmostEqual(scores[7], 1)
        self.assertAlmostEqual(scores[-1], -1)

    def test_body_graph_requires_integral_width_and_order(self):
        markers = [(57, 8648 + 2048), (70, 8648 + 2048 + 4 * 4096), (92, 8648 + 2049)]
        results = body_paths(markers, 8420)
        self.assertEqual(results, [(2, ((57, 1),)), (2, ((57, 1), (70, 4)))])

    def test_crc_samples_never_change_structure_selection(self):
        layout = (2, ((57, 1), (70, 4)))
        flags = header_candidates(layout)
        info = word_bits(0x34449C80).tolist() + word_bits(0x06220000).tolist()
        info = np.array(info, dtype=np.uint8)
        expected = crc8(info[:-8])
        info[-8:] = word_bits(expected)[-8:]
        first = (1 - 2 * coded_bits(info).astype(float)).astype(complex)
        info[-8:] = word_bits(expected ^ 16)[-8:]
        second = (1 - 2 * coded_bits(info).astype(float)).astype(complex)
        for a in flags:
            self.assertEqual(information_score(first, a), information_score(second, a))
        winner = max(flags, key=lambda a: information_score(first, a))
        self.assertTrue(check_crc_tail(first, winner)['crc_ok'])
        self.assertFalse(check_crc_tail(second, winner)['crc_ok'])
