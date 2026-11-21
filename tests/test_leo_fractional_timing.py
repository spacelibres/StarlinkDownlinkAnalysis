from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E089/scripts"))
from audit_leo_fractional_timing import common_aperture, tone_atoms


class LeoFractionalTimingTests(unittest.TestCase):
    def test_common_samples_retain_labels_and_only_rotate_tone_columns(self):
        fs = 2_500_000
        offsets = np.array([0., 1/3, -1/3])
        index, slot, local = common_aperture(541, 1.25 + offsets, fs, 4.4e-6, 2/15e6, 1.5)
        frequencies = np.array([-820312.5, 117187.5, 351562.5])
        self.assertTrue(len(index) > 0)
        self.assertEqual((int(slot.min()), int(slot.max())), (32, 48))
        self.assertTrue(np.all(np.diff(index) > 0))
        phases = np.exp(-2j * np.pi * offsets[:, None] * frequencies / fs)
        np.testing.assert_allclose(
            tone_atoms(local, frequencies, 2/15e6),
            tone_atoms(local[:1], frequencies, 2/15e6) * phases[:, None, :],
            atol=2e-12, rtol=0,
        )
