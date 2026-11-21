import unittest

import numpy as np

import project_paths
from audit_mcs_lower_rate_core import BIT_LENGTH, right_aligned, transmitted_system_rows


class LowerRateCoreTests(unittest.TestCase):
    def test_right_alignment_preserves_high_parity_and_erases_negative_source_coordinates(self):
        rows = np.array([[0, 1279, 1280, 12287, 12288, 16383]])
        wanted = [[BIT_LENGTH, BIT_LENGTH, 0, 11007, 12288, 16383]]
        np.testing.assert_array_equal(right_aligned(rows, 11008), wanted)

    def test_encoder_support_ignores_shortened_zero_coordinates(self):
        rows = np.array([[0, BIT_LENGTH, 12288], [BIT_LENGTH, 11007, 16383]])
        supports = transmitted_system_rows(rows, 11008)
        self.assertEqual([a.tolist() for a in supports], [[0], [11007]])
