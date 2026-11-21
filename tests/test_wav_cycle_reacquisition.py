import unittest

import numpy as np

import project_paths
from audit_wav_cycle_reacquisition import profile_delay, complex_tone_grid


class WavCycleReacquisitionTests(unittest.TestCase):
    def test_delay_search_refits_complex_gain_and_finds_shift(self):
        count = 64
        bins = np.array([-19, -13, -3, 2, 7, 15])
        columns = np.exp(1j * .3 * bins)[None, :, None]
        gain = np.array([2 + 1j])
        samples = (columns[0] @ gain) * np.exp(-2j * np.pi * bins * 21 / count)
        phase, delay, found, _ = profile_delay(columns, samples, bins, count)
        self.assertEqual((phase, delay), (0, 21))
        np.testing.assert_allclose(found, gain, atol=1e-12)

    def test_centered_fft_interpolation_preserves_complex_tone(self):
        fs, count = 1024, 256
        times = (np.arange(count) - (count - 1) / 2) / fs
        signal = (1 + 2j) * np.exp(2j * np.pi * 71.25 * times)
        received = complex_tone_grid(signal, fs, np.array([71.25]))
        np.testing.assert_allclose(received, [1 + 2j], atol=.02)
