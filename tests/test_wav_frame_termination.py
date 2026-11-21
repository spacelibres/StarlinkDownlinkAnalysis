from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E090/scripts"))
from audit_wav_frame_termination import step_bank, step_fit


class WavFrameTerminationTests(unittest.TestCase):
    def test_step_line_profile_recovers_known_offsets(self):
        time = 13.4 + np.arange(351) / 750
        centers = np.array([171, 172, 174])
        candidates = np.arange(150, 200)
        regressor, trend, contrasts = step_bank(time, time, candidates)
        y = 2 + .2 * (time - time.mean())[:, None] + 4 * (time[:, None] < time[centers])
        outputs, _ = step_fit(y, regressor, trend, contrasts)
        np.testing.assert_array_equal(candidates[outputs], centers)
