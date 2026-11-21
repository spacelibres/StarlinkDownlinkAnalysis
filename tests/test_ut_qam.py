import unittest

import numpy as np

import project_paths
from extract_ut_qam import allocation_regions
from audit_ut_qam_fec import COMPLEMENTS, complement_masks, high_margin_checks
from qc_figure_core import qc_syndrome


class UTQAMTests(unittest.TestCase):
    def test_word_regions_follow_all_descriptors_and_header(self):
        header = {"predicted_h2_start": 1000, "h2": {"blocks": 4,
                  "layout20": {"entries": [{"mcs_candidate": 57, "quantity": 2},
                                          {"mcs_candidate": 66, "quantity": 3}]}}}
        self.assertEqual(allocation_regions(header, {57: 20, 66: 40}),
                         [(57, 1456, 1496, 2), (66, 1496, 1616, 3)])

    def test_lane_complement_masks_match_explicit_syndromes(self):
        shifts = np.array([[0, 3, -1, 6], [-1, 2, 4, 5]])
        patterns = np.tile(COMPLEMENTS, (1, 8))
        syndrome = qc_syndrome(patterns, shifts, 8).reshape(16, 2, 2, 4)
        np.testing.assert_array_equal(syndrome[:, :, 0], complement_masks(shifts))
        np.testing.assert_array_equal(syndrome[:, :, 1], complement_masks(shifts))

    def test_weak_symbol_removes_only_incident_checks(self):
        shifted = np.array([[0, 0]])
        bits = np.zeros((1, 16), dtype=np.uint8)
        margins = np.array([[.1, .01, .1, .1]])
        syndrome, mask = high_margin_checks(bits, margins, shifted, 8)
        self.assertFalse(syndrome.any())
        self.assertEqual(mask.tolist(), [[True, True, True, True, False, False, False, False]])


if __name__ == "__main__":
    unittest.main()
