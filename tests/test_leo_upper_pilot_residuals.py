import unittest

import numpy as np

import project_paths
from audit_leo_upper_pilot_residuals import pilot_projections, phase_gain


class LeoPilotPhaseTransferTests(unittest.TestCase):
    def test_common_residual_phase_transfers_across_disjoint_tones(self):
        rng = np.random.default_rng(21)
        times = np.arange(150) * 8.8e-6
        gains = rng.normal(size=8) + 1j * rng.normal(size=8)
        cfo = 187.0
        phase = np.zeros(len(times))
        phase[1::2] = rng.choice([-0.9, 0.9], size=75)
        matrix = np.exp(1j * (2 * np.pi * cfo * times + phase))[:, None] * gains
        first, second = pilot_projections(matrix, times, cfo)
        self.assertAlmostEqual(phase_gain(first, second), 1 - np.cos(0.9))
        self.assertLess(phase_gain(first, second, shift=3), 0.15)
        self.assertLess(phase_gain(first, second, invert=True), 0)

    def test_constant_phase_does_not_invent_a_transfer_gain(self):
        times = np.arange(150) * 8.8e-6
        matrix = np.broadcast_to(np.exp(1j * np.arange(8)), (150, 8))
        first, second = pilot_projections(matrix, times, 0.0)
        self.assertAlmostEqual(phase_gain(first, second), 0.0)
