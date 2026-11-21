import unittest

import numpy as np

import project_paths
from audit_legacy_pn_transfer import longest_gain, recurrence_residual
from receiver_models import pn_bits


class LegacyPnTransferTests(unittest.TestCase):
    def test_lag_annihilates_periodic_mask_without_fitting_pn_phase(self):
        rng = np.random.default_rng(19)
        motif = rng.integers(0, 2, 24, dtype=np.uint8)
        indices = np.arange(480)
        observed = pn_bits()[(indices + 230) % 32767] ^ motif[indices % 24]
        self.assertFalse(recurrence_residual(observed, 14).any())
        self.assertTrue(recurrence_residual(observed, 13).any())

    def test_gain_coordinates_exclude_the_right_boundary(self):
        self.assertEqual(longest_gain(np.array([-2, 1, 1, 2, -3, -2])), (1, 4, 4.0))


if __name__ == "__main__":
    unittest.main()
