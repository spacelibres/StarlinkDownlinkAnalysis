import unittest

import numpy as np

import project_paths
from audit_wav_slot_frame_power import extract_frames, paired_moments, stable_signs
from verify_wav_slot_sign_transfer import evaluate_predictions


class WavSlotFramePowerTests(unittest.TestCase):
    def test_frame_transform_corrects_fractional_start_and_carrier_offset(self):
        fs = 2_400_000
        grid = 750.0
        harmonics = np.array([-33, 87])
        amplitudes = np.array([1.2 - .2j, .7j])
        time = (np.arange(32_000) - 15_999.5) / fs
        lag, offset = 4.375, -68.0
        cycles = time * grid - lag / 3200
        signal = (np.exp(2j * np.pi * cycles[:, None] * harmonics) @ amplitudes) * np.exp(2j * np.pi * offset * time)
        _, observed = extract_frames(signal, time, fs, grid, offset, lag, harmonics)
        np.testing.assert_allclose(observed, np.broadcast_to(amplitudes, observed.shape), atol=1e-11)

    def test_zero_average_switching_keeps_cross_partition_power(self):
        alternating = np.array([1., -1., 1., -1.])
        stable = np.ones(4)
        self.assertEqual(paired_moments(alternating, alternating)['centered_real_power'], 1.)
        self.assertEqual(paired_moments(stable, stable)['centered_real_power'], 0.)
        self.assertEqual(paired_moments(alternating, -alternating)['centered_real_power'], -1.)

    def test_symbol_consensus_requires_four_agreeing_projections(self):
        observed = np.asarray([[.9, -.9, .6, .7], [.8, -.8, .1, -.6],
                               [.9, -.8, .8, .7], [.9, -.7, .9, .7]])
        self.assertEqual(stable_signs(observed).tolist(), [1, -1, 0, 0])

    def test_donor_sign_is_fixed_before_target_scoring(self):
        first = np.asarray([1., -1., .2, -.8])
        second = np.asarray([1., -1., .9, .9])
        target = np.array([.8, -1.2, 1., 1.])
        forward = evaluate_predictions(first, second, target, np.array([2]))
        reversed_ = evaluate_predictions(first, second, -target, np.array([2]))
        self.assertEqual(forward['eligible'], 2)
        self.assertEqual(forward['eligible'], reversed_['eligible'])
        self.assertAlmostEqual(forward['mean_target_projection'], 1)
        self.assertAlmostEqual(reversed_['mean_target_projection'], -1)
