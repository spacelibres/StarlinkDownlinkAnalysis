import unittest

import numpy as np

import project_paths
from audit_overlap_prefix_cancellation import (
    calibration_fft, fft_gain, pilot_features, predict, CP, FFT_SIZE, RATE, SLOT, SLOTS,
)


class OverlapPrefixTests(unittest.TestCase):
    def test_same_grid_prefix_samples_repeat(self):
        positions = np.arange(20, 61) + 30*SLOT
        np.testing.assert_allclose(pilot_features(positions, 0., 1., 0.),
                                   pilot_features(positions+FFT_SIZE, 0., 1., 0.), atol=1e-11)
        left = pilot_features(positions, 400., 1., 0.)
        right = pilot_features(positions+FFT_SIZE, 400., 1., 0.)
        self.assertGreater(np.linalg.norm(left-right), 1.)

    def test_other_grid_pilots_survive_subtraction_and_freeze_on_holdout(self):
        times = np.arange(71, 159)[:, None]*SLOT + np.arange(9, 50, 2)
        a = pilot_features(times, 0., 1., 0.)
        a_tail = pilot_features(times+FFT_SIZE, 0., 1., 0.)
        b = pilot_features(times, 499., 1., 631.)
        b_tail = pilot_features(times+FFT_SIZE, 499., 1., 631.)
        rng = np.random.default_rng(59)
        gains = rng.normal(size=(2, 16)) + 1j*rng.normal(size=(2, 16))
        measured = (5*(a_tail-a)) @ gains[0] + (b_tail-b) @ gains[1]
        train = np.arange(len(times)) % 2 == 0
        result = predict(measured, b_tail-b, train)
        self.assertAlmostEqual(result['improvement'], 1.)
        self.assertAlmostEqual(result['coherence'], 1.)

    def test_fft_calibration_tracks_the_relative_carrier_phase(self):
        epoch = 400
        frequencies = (13_700., -21_900.)
        slots = np.arange(2, 213)
        points = epoch+slots[:, None]*SLOT+np.arange(SLOT)
        gains = np.exp(.2j*np.arange(16))
        base = pilot_features(points, epoch, 1., 0.) @ gains
        y = np.zeros(int(points.max()+10), dtype=complex)
        y[points] = base*np.exp(2j*np.pi*frequencies[0]*points/RATE)
        observed, used_slots = calibration_fft(y, epoch, 1., frequencies[0], 0., 1., frequencies[1])
        estimated = fft_gain(observed, used_slots, np.ones(len(SLOTS), dtype=bool))
        phase = np.exp(2j*np.pi*(frequencies[0]-frequencies[1])*epoch/RATE)
        np.testing.assert_allclose(estimated, gains*phase, atol=1e-11)
