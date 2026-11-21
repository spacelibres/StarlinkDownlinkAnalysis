from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E093/scripts"))
from audit_leo_copy_transfer import POSITIONS, record_roles, tally


class LeoCopyTransferTests(unittest.TestCase):
    def test_fixed_air_coordinates_change_record_role_with_length(self):
        self.assertEqual(POSITIONS, (752, 758))
        self.assertEqual(record_roles(36, [459], 14), [(464, "trajectory_459"), (470, "trajectory_459")])
        self.assertEqual(record_roles(40, [459], 14), [(432, "frame_count"), (438, "frame_count")])

    def test_erasure_does_not_enter_observed_agreement(self):
        rows = [{"equal": True, "observed_both": True}, {"equal": False, "observed_both": False}]
        self.assertEqual(tally(rows), {"pairs": 2, "equal": 1, "observed_pairs": 1, "observed_equal": 1})
