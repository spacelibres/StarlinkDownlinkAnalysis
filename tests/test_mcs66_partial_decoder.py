from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E088/scripts"))
from audit_mcs66_partial_decoder import constellation_llr, min_sum, output_decisions, syndrome_count


class PartialMcs66DecoderTests(unittest.TestCase):
    def test_single_constraint_corrects_a_weak_wrong_sign(self):
        support = np.array([[0, 1, 2]])
        llr = np.array([[8.0, -8.0, .2], [-8.0, -8.0, -.2]])
        result = min_sum(llr, support, .8, 1)
        np.testing.assert_array_equal(result < 0, [[False, True, True], [True, True, False]])
        np.testing.assert_array_equal(syndrome_count(result < 0, support), [0, 0])

    def test_llr_sign_is_positive_for_bit_zero(self):
        points = np.array([1+1j, -1+1j, 1-1j, -1-1j])
        mapping = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        scores = constellation_llr(points[None], points, mapping, .05, log_map=True)
        np.testing.assert_array_equal(scores.reshape(-1, 2) < 0, mapping)

    def test_zero_posterior_keeps_observed_values_and_abstains_on_erasure(self):
        bits, seen = output_decisions(
            np.array([[0., 0., -0.8, 0.8]]),
            np.array([[True, False, False, True]]),
            np.array([[True, False, False, False]]),
        )
        np.testing.assert_array_equal(bits, [[True, False, True, False]])
        np.testing.assert_array_equal(seen, [[True, False, True, True]])
