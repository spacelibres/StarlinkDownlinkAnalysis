from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E097/scripts"))
from audit_code5_clock_correction import CLOCK_BITS, projected_clock, timestamp_numerator


class Code5ClockCorrectionTests(unittest.TestCase):
    def test_rebuilds_timestamp_using_only_disjoint_narrowband_bits(self):
        base = 1421546970
        count, correction = 5540, 3374756
        record = (count << 421) | (correction << 453) | ((2 * base + 1) << 485)
        data = bytes([32]) + bytes(31) + record.to_bytes(96, "little")
        seen = np.zeros(1024, bool)
        seen[[32 * 8 + i for i in CLOCK_BITS]] = True
        self.assertEqual(projected_clock(data, seen, 32), (base, count, correction))
        self.assertEqual(timestamp_numerator(base, count, correction) // 27, 1421546977389157118)
        seen[32 * 8 + 519] = False
        self.assertEqual(projected_clock(data, seen, 32), (base, count, correction))
        seen[32 * 8 + 453] = False
        self.assertIsNone(projected_clock(data, seen, 32))

    def test_integer_time_handles_the_recorded_older_odd_correction(self):
        gps_ns = timestamp_numerator(1347023220, 4333, 4666571) // 27
        self.assertEqual(gps_ns, 1347023225777431535)
