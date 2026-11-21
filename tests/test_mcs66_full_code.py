import unittest

import numpy as np
import project_paths
from recover_mcs66_full_code import parity_blocks
from validate_mcs66_full_code import invert_binary


class Mcs66FullCodeTests(unittest.TestCase):
    def test_parity_motif_has_low_pivot_and_three_observed_high_choices(self):
        self.assertEqual(parity_blocks(0), (0, 4, 5, 6))
        self.assertEqual(parity_blocks(1), (1, 5, 6, 7))
        self.assertEqual(parity_blocks(2), (2, 4, 6, 7))
        self.assertEqual(parity_blocks(3), (3, 4, 5, 7))

    def test_binary_inverse_pivots_over_gf2_and_rejects_singular_matrix(self):
        a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        inverse = np.array([[p >> i & 1 for i in range(3)] for p in invert_binary(a)])
        np.testing.assert_array_equal(a @ inverse % 2, np.eye(3, dtype=int))
        with self.assertRaisesRegex(ValueError, "singular"):
            invert_binary(np.ones((2, 2), dtype=np.uint8))
