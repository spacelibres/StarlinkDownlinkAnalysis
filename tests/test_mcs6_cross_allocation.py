from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_mcs6_cross_allocation import ordered_boundaries


class MCS6CrossAllocationTests(unittest.TestCase):
    def test_failed_suffix_does_not_erase_earlier_predictions(self):
        prefix, complete = ordered_boundaries([3, 7], [(1, 5), (4, 4)])
        self.assertEqual(prefix, [[(0, 0, 0), (0, 1, 3)], []])
        self.assertEqual(complete, [[], []])

    def test_complete_paths_preserve_zero_and_all_lengths_not_just_first(self):
        prefix, complete = ordered_boundaries([2, 5, 6], [(1, 8), (2, 6)])
        self.assertEqual(prefix[0], [(0, 0, 0), (0, 1, 2), (0, 2, 7)])
        self.assertEqual(complete, [[(0, 1, 2)], [(1, 3, 11)]])
