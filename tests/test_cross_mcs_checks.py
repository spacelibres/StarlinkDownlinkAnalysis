import unittest

import numpy as np
import project_paths
from audit_cross_mcs_checks import paired_syndromes, visible_terms


class CrossMcsChecksTests(unittest.TestCase):
    def test_wrong_serial_origin_uses_extra_bit_and_common_observed_support(self):
        bits = np.array([[0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=np.uint8)
        masks = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=bool)
        support = np.array([[0, 4], [1, 2]])
        correct, shifted = paired_syndromes(bits, masks, np.array([7, 8]), support, support + 1)
        self.assertEqual((correct["seen"], correct["violations"]), (3, 2))
        self.assertEqual((shifted["seen"], shifted["violations"]), (3, 2))
        self.assertEqual([row["seen"] for row in correct["groups"]], [2, 1])

    def test_visibility_maps_multiple_lanes_and_wraps_to_next_symbol(self):
        support = np.array([[0, 4, 5, 15], [5, 10, 14, 19]])
        visible = np.array([True, False, True, True])
        np.testing.assert_array_equal(visible_terms(3, support, 5, visible), [4, 2])
