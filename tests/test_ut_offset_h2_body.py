import unittest

import numpy as np

import project_paths
from verify_ut_offset_h2_body import matched_pn, predicted_markers


class UtOffsetH2BodyTests(unittest.TestCase):
    def test_marker_is_systematic_end_of_final_word(self):
        entries = [{'mcs_candidate': 70, 'quantity': 2}, {'mcs_candidate': 92, 'quantity': 1}]
        first, second = predicted_markers(1000, entries)
        self.assertEqual(first['word_stop'], 1000 + 2 * 4096)
        self.assertEqual(first['full_systematic_stop'], first['word_stop'] - 4096 + 3200)
        self.assertEqual(second['systematic_mixed_skip'], 1)
        self.assertEqual(second['full_systematic_stop'], first['word_stop'] + 2611)

    def test_matched_filter_preserves_signed_complex_phase(self):
        pattern = np.array([1, 1j, -1, -1j])
        windows = np.stack([pattern, -pattern, 1j * pattern, np.zeros(len(pattern))])
        np.testing.assert_allclose(matched_pn(windows, pattern), [1, -1, 0, 0])
