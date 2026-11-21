import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import project_paths
sys.path.insert(0, str(ROOT / 'experiments/E015/scripts'))
from code9_core import edge_slope_lifts, systematic_checks, split_frames, parity_disagreements


class Code9ValidationTests(unittest.TestCase):
    def test_cross_edge_phase_lifts_follow_time_and_keep_the_low_edge_anchor(self):
        lower, upper = -492, 492
        true_edge_phase = np.linspace(0.4, 4 * np.pi, 31)
        wrapped = np.angle(np.exp(1j * true_edge_phase))
        phase, lifts = edge_slope_lifts(wrapped / (upper - lower), np.array([lower, 0, upper]), lower, upper)
        np.testing.assert_allclose(phase[:, 0], 0)
        np.testing.assert_allclose(phase[:, 2] + wrapped, true_edge_phase)
        np.testing.assert_allclose(phase[:, 1], np.pi * lifts)
        self.assertEqual(int(lifts[-1]), 2)

    def test_split_preserves_whole_frames_and_fixed_assignment(self):
        frames = np.repeat([0, 4, 8, 12, 16], [1, 7, 10, 7, 6])
        training, order = split_frames(frames, seed=2026112015, train_groups=3)
        self.assertEqual(order, [16, 8, 12, 4, 0])
        np.testing.assert_array_equal(training, np.isin(frames, [8, 12, 16]))
        self.assertEqual(int(np.sum(training)), 23)

    def test_rectangular_systematic_model_rejects_injected_parity_errors(self):
        p = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
        u = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        v = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        words = np.c_[u, v]
        h = systematic_checks(p)
        np.testing.assert_array_equal((words @ h.T) % 2, np.zeros((2, 2)))
        words[1, -1] ^= 1
        np.testing.assert_array_equal(parity_disagreements(words, p), [0, 1])


if __name__ == '__main__':
    unittest.main()
