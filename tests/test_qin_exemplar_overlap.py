import unittest

import project_paths
from audit_qin_exemplars import consistent_clock_pairs


class QinOverlapTests(unittest.TestCase):
    def test_shared_clock_rejects_unrelated_duplicate_headers(self):
        pairs = [
            {"qin_frame": 249, "ut_candidate": 0, "toa_offset_s": 4.8788},
            {"qin_frame": 249, "ut_candidate": 4, "toa_offset_s": 4.8559},
            {"qin_frame": 252, "ut_candidate": 4, "toa_offset_s": 4.878800001},
        ]
        _, result = consistent_clock_pairs(pairs, 1e-6)
        self.assertEqual([(r["qin_frame"], r["ut_candidate"]) for r in result], [(249, 0), (252, 4)])

    def test_same_frame_cannot_match_two_candidates(self):
        pairs = [
            {"qin_frame": 249, "ut_candidate": 0, "toa_offset_s": 4.8788},
            {"qin_frame": 249, "ut_candidate": 4, "toa_offset_s": 4.878800001},
        ]
        with self.assertRaises(ValueError):
            consistent_clock_pairs(pairs, 1e-6)


if __name__ == "__main__":
    unittest.main()
