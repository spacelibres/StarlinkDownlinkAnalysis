import unittest

import numpy as np

import project_paths
from audit_leo_pilot_phase_transfer import compare_phase, profile_pilots


class LeoPilotPhaseTests(unittest.TestCase):
    def test_phase_predictions_respect_wrapping_and_find_local_rotation(self):
        smooth = np.exp(1j*(2.8 + .43*np.arange(8)))
        weights = np.ones(8)
        plain = compare_phase(smooth, smooth, weights)
        np.testing.assert_allclose(plain['smooth_error'], 0, atol=1e-12)
        np.testing.assert_allclose(plain['global_smooth_error'], 0, atol=1e-12)
        shifted = smooth.copy()
        shifted[0] *= np.exp(.85j)
        local = compare_phase(shifted, shifted, weights)
        self.assertAlmostEqual(local['phase_delta_rad'][0], .85)
        self.assertAlmostEqual(local['global_phase_delta_rad'][0], .85)
        self.assertGreater(local['smooth_error'][0], .6)
        self.assertAlmostEqual(local['individual_error'][0], 0.)
        self.assertGreater(local['reversed_error'][0], local['smooth_error'][0])

    def test_frequency_gain_fit_depends_only_on_training_samples(self):
        rng = np.random.default_rng(2026112164)
        time = np.arange(360) / 250_000
        design = np.exp(.5j*np.pi*rng.integers(0, 4, (time.size, 8)))
        gains = np.exp(.4j*np.arange(8))
        signal = (design @ gains)*np.exp(2j*np.pi*time*175.)
        calibration = np.arange(time.size) % 3 < 2
        initial = profile_pilots(design, signal, time, calibration)
        signal[~calibration] *= -1.
        altered = profile_pilots(design, signal, time, calibration)
        self.assertEqual(initial['cfo_hz'], 175.)
        self.assertEqual(altered['cfo_hz'], 175.)
        np.testing.assert_allclose(initial['calibration_gain'], gains)
        np.testing.assert_allclose(initial['calibration_gain'], altered['calibration_gain'])
        np.testing.assert_allclose(altered['held_gain'], -gains)
