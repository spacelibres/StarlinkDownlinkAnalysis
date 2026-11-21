import unittest

import numpy as np

import project_paths
from audit_ut_weak_padding_and_parity import onset_profile, parity_prediction, qam_likelihood


class WeakPaddingParityTests(unittest.TestCase):
    def test_padding_cutoff_resolves_inside_a_symbol(self):
        labels = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        points = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        rx = np.array([points[3], points[2], points[0]])
        logp = qam_likelihood(rx, np.ones(3), points, .01)
        scores = onset_profile(logp, labels, np.zeros(6, int), 5, np.ones(3, bool))
        self.assertEqual(scores.argmax(), 3)

    def test_whole_target_symbol_is_reserved(self):
        rows = np.array([[0, 1, 4, 6], [1, 4, 6, 7]])
        moments = np.zeros((2, 16))
        moments[0, [3, 2]] = [.9, -.7]
        moments[1, [5, 13]] = [.2, -.3]
        predicted, observed = parity_prediction(moments, rows, 4)
        np.testing.assert_allclose(predicted, [.9, -.7])
        np.testing.assert_allclose(observed, [.2, -.3])
