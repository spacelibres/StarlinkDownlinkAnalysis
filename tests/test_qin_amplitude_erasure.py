from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E075/scripts"))
from audit_qin_amplitude_erasure import corner_slots, holdout_error


class QinAmplitudeErasureTests(unittest.TestCase):
    def test_corner_test_requires_all_points_and_noncorner_peer(self):
        left = np.ones((3, 4), complex)
        right = left.copy()
        right[0, 0] = 1 / 3
        left[2, 0] = 1 / 3
        np.testing.assert_array_equal(corner_slots(left, right), [True, False, True])

    def test_holdout_gain_uses_only_calibration_positions(self):
        ref = np.ones(4, complex)
        measured = np.array([1, 1, 3, 3], complex)
        self.assertAlmostEqual(holdout_error(measured, ref, np.array([True, True, False, False])), 4 / 9)
