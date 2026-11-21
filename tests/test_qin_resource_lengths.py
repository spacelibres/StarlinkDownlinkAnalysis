import unittest

import numpy as np

import project_paths
from audit_qin_lengths import terminal_edge


class TerminalEdgeTests(unittest.TestCase):
    def test_phase_recovers_boundary_and_global_rotation(self):
        rng = np.random.default_rng(37)
        template = 1. - 2. * rng.integers(0, 2, 60)
        values = rng.standard_normal(2400) + 1j * rng.standard_normal(2400)
        first = 1618
        indices = np.arange(len(values) - first)
        values[first:] = (.4 - .7j) * template[indices % 60]
        self.assertEqual(terminal_edge(values, template)["phase_refined_edge"], first)

    def test_noise_is_rejected_by_terminal_coherence(self):
        rng = np.random.default_rng(81)
        template = 1. - 2. * rng.integers(0, 2, 60)
        values = rng.standard_normal(2400) + 1j * rng.standard_normal(2400)
        self.assertIsNone(terminal_edge(values, template)["phase_refined_edge"])


if __name__ == "__main__":
    unittest.main()
