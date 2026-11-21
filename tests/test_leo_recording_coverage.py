from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E071/scripts"))
from audit_leo_recording_coverage import carrier_if_hz, coverage, signed_bin


class LeoRecordingCoverageTests(unittest.TestCase):
    def test_dft_wrap_cannot_be_used_as_rf_distance(self):
        self.assertEqual([signed_bin(k) for k in (511, 512, 528, 535)], [511, -512, -496, -489])
        lower = (carrier_if_hz(4, 528) + carrier_if_hz(4, 535)) / 2
        upper = (carrier_if_hz(4, 488) + carrier_if_hz(4, 495)) / 2
        self.assertEqual(lower, 1_709_687_500)
        self.assertEqual(upper, 1_940_312_500)
        self.assertEqual(carrier_if_hz(4, 511) - lower, 235_195_312.5)

    def test_native25_target_has_limited_guard(self):
        row = coverage(4, 1_932_500_000, 25_000_000, 25_000_000)
        self.assertEqual(row["target_baseband_hz"], 12_382_812.5)
        self.assertEqual(row["nominal_guard_hz"], 117_187.5)
        self.assertEqual(coverage(4, 1_932_500_000, 25_000_000, 24_000_000)["nominal_guard_hz"], -382_812.5)
        self.assertEqual(coverage(4, 1_717_500_000, 25_000_000, 25_000_000)["nominal_guard_hz"], -214_882_812.5)
