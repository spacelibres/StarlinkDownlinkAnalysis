import itertools
import unittest

import project_paths
from audit_leo_native_clock_coverage import complete_anchor, missing_clock_bits


class LeoNativeClockCoverageTests(unittest.TestCase):
    def test_frequency_reference_and_edge_are_not_folded(self):
        self.assertEqual(missing_clock_bits(0, 12_382_812.5), ())
        self.assertEqual(missing_clock_bits(350_000, 12_382_812.5), (519,))
        self.assertEqual(missing_clock_bits(360_000, 12_382_812.5), (518, 519))

    def test_two_top_anchor_bits_are_unique_in_all_residue_classes(self):
        missing = (518, 519)
        mask = sum(1 << bit for bit in missing)
        for top, base in itertools.product(range(4), range(0, 225, 15)):
            seconds = top * (1 << 32) + base
            seconds -= seconds % 15
            packed = ((seconds * 2 + 1) << 485) | (1573 << 421)
            self.assertEqual(complete_anchor(packed ^ mask, missing), ((seconds, 1573, 0),))

    def test_fail_closed_on_invalid_visible_bits_and_preserve_ambiguity(self):
        packed = (1 << 485) | (12 << 421)
        self.assertEqual(complete_anchor(packed ^ (1 << 486), (518, 519)), ())
        self.assertEqual(len(complete_anchor(packed, (516, 517, 518, 519))), 2)
        self.assertEqual(complete_anchor(packed, (421,)), ())
