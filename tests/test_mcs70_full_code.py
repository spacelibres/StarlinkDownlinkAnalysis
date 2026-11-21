import unittest

import numpy as np

import project_paths
from recover_mcs70_full_code import BIT_LENGTH, append_tap, lift_family, xor_families


class Mcs70FullCodeTests(unittest.TestCase):
    def test_lift_preserves_parity_coordinates_and_moves_the_omitted_block(self):
        rows = np.array([[0, 12287, 12288, 12799, 12800, 16383]])
        expected = [[512, 12799, BIT_LENGTH, BIT_LENGTH + 511, 12800, 16383]]
        np.testing.assert_array_equal(lift_family(rows), np.sort(expected, axis=1))

    def test_xor_cancels_only_shared_coordinates(self):
        rows = xor_families(np.array([[12, BIT_LENGTH], [13, BIT_LENGTH + 1]]),
                            np.array([[18, BIT_LENGTH], [19, BIT_LENGTH + 1]]))
        np.testing.assert_array_equal(rows, [[12, 18], [13, 19]])

    def test_early_tap_rotates_in_the_first_block(self):
        rows = np.repeat([[512, 8192]], 512, axis=0)
        fixed = append_tap(rows, 511)
        np.testing.assert_array_equal(fixed[:2], [[511, 512, 8192], [0, 512, 8192]])
        np.testing.assert_array_equal(append_tap(rows, None), rows)
