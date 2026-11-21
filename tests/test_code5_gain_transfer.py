from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E081/scripts"))
from audit_code5_gain_transfer import NONPILOT, reference_axis, shared_counts, visible_lanes


class Code5GainTransferTests(unittest.TestCase):
    def test_gain_uses_only_supported_signed_samples(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        observed = np.array([True, True, True, False])
        source = np.array([1 + .1j, -1 - .1j, -1 - .1j, 90 + 5j])
        self.assertAlmostEqual(reference_axis(source, bits, observed), 1 + .1j)

    def test_disagreements_use_only_jointly_observed_bits(self):
        left = np.array([[0, 1, 1], [1, 1, 1]], dtype=np.uint8)
        right = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        first = np.ones_like(left, dtype=bool)
        second = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
        known, errors = shared_counts(left, right, first, second)
        self.assertEqual((known.tolist(), errors.tolist()), ([2, 2], [1, 2]))

    def test_corner_erasure_crosses_slot_boundary_without_removing_signs(self):
        mask = visible_lanes(np.array([True, True, False]), len(NONPILOT) - 1,
                             np.array([False, True]), np.array([True, False]))
        np.testing.assert_array_equal(mask, [[True, False], [True, True], [False, False]])
