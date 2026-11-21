from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E098/scripts"))
from audit_code5_narrow_waveform_clock import FS, WIDE_FS, fit_counter_origin, narrow_waveform


class Code5NarrowWaveformClockTests(unittest.TestCase):
    def test_25_msps_projection_rejects_outside_tone(self):
        t = np.arange(48000) / WIDE_FS
        center = 100_000_000
        surviving = 2_500_000
        iq = np.exp(2j * np.pi * (center + surviving) * t) + np.exp(2j * np.pi * (center + 14_000_000) * t)
        narrow = narrow_waveform(iq, center, 0)
        expected = np.exp(2j * np.pi * surviving * np.arange(len(narrow)) / FS)
        self.assertEqual(len(narrow), 2500)
        np.testing.assert_allclose(narrow, expected, atol=2e-10, rtol=0)

    def test_counter_origin_uses_high_bits_and_rejects_reversed_cadence(self):
        ordinals = [0, 1, 3, 5, 6]
        high = [(4333 + i) // 2 for i in ordinals]
        self.assertEqual(fit_counter_origin(high, ordinals), 4333)
        self.assertIsNone(fit_counter_origin(high, ordinals[::-1]))
