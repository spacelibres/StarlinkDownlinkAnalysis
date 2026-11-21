from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E095/scripts"))
from audit_code5_gps_epoch import integer_at, split_timestamp, utc_label


class Code5GpsEpochTests(unittest.TestCase):
    def test_unaligned_seconds_and_nanoseconds_across_carry(self):
        s = 1421546977
        a = (s << 348) | (999_999_999 << 380) | (0x7FF << 410)
        b = ((s + 1) << 348) | (0x7FF << 410)
        self.assertEqual(split_timestamp(a), (s, 999_999_999))
        self.assertEqual(split_timestamp(b), (s + 1, 0))
        self.assertEqual(integer_at(a, 410, 11), 0x7FF)
        self.assertEqual(utc_label(s, 389_157_118), "2025-01-22T02:09:19.389157118Z")

    def test_rejects_non_canonical_fraction(self):
        with self.assertRaisesRegex(ValueError, "nanoseconds"):
            split_timestamp(1_000_000_000 << 380)
