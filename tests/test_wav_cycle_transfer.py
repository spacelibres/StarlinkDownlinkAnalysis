import unittest

import numpy as np

import project_paths
from audit_wav_cycle_transfer import fit_tones, select_frame_ids


class WavCycleTests(unittest.TestCase):
    def test_fit_separates_moving_tone_and_fixed_dc(self):
        frequencies = np.array([-8e5, 5e5, 0, 1.8e5])
        times = np.arange(10) / 2.4e6
        coefficients = np.array([1 + .5j, .7 - 2j, 17 + 3j, -7j])
        signal = np.exp(2j * np.pi * times[:, None] * frequencies) @ coefficients
        estimates, _ = fit_tones(signal[None, :], frequencies, 2.4e6)
        np.testing.assert_allclose(estimates[0], coefficients, atol=1e-11)

    def test_frame_selection_is_unique_and_reproducible(self):
        a = select_frame_ids(10, 1500, 1650)
        b = select_frame_ids(10, 1500, 1650)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a.tolist())), 16)
        self.assertTrue(np.all((a >= 1500) & (a < 1650)))


if __name__ == "__main__":
    unittest.main()
