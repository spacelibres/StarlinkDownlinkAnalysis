from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E072/scripts"))
from audit_leo_edge_passband import cut_and_restore, lowpass_gain


class LeoEdgePassbandTests(unittest.TestCase):
    def test_taper_and_nyquist_boundary(self):
        freq = np.array([-12.5e6, -12.375e6, 12.25e6, 12.5e6, 12.75e6])
        np.testing.assert_allclose(lowpass_gain(freq, 25e6, 250e3), [0, .5, 1, 0, 0])
        np.testing.assert_array_equal(lowpass_gain(freq, 25e6, 0), [0, 1, 1, 0, 0])

    def test_actual_decimation_and_interpolation_reject_out_of_band_tone(self):
        t = np.arange(10_000) / 250e6
        in_band = np.exp(2j * np.pi * 12.375e6 * t)
        out_of_band = np.exp(2j * np.pi * 12.625e6 * t)
        restored = cut_and_restore(in_band + out_of_band, 25e6, 250e3)
        np.testing.assert_allclose(restored, .5 * in_band, atol=2e-11)
