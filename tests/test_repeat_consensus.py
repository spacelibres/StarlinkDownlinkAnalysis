from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E076/scripts"))
from audit_repeat_consensus import audit_code5_triplet, compare_section, fuse_two


class RepeatConsensusTests(unittest.TestCase):
    def test_conflicting_and_missing_positions_remain_erased(self):
        values = np.array([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]], dtype=np.uint8)
        masks = np.array([[True, True, True, False, False], [True, True, False, True, False]])
        predicted, agreed, single, conflict = fuse_two(values, masks)
        np.testing.assert_array_equal(predicted[agreed | single], [0, 1, 1])
        np.testing.assert_array_equal(agreed, [True, False, False, False, False])
        np.testing.assert_array_equal(single, [False, False, True, True, False])
        np.testing.assert_array_equal(conflict, [False, True, False, False, False])

    def test_reference_does_not_fill_unknown_donors(self):
        values = np.array([[1, 1, 1], [1, 0, 1]], np.uint8)
        masks = np.array([[True, True, False], [True, True, False]])
        held = np.array([0, 1, 1], np.uint8)
        row = compare_section(values, masks, held, np.ones(3, bool))
        self.assertEqual(row["agreed_both"], [1, 1])
        self.assertEqual(row["single_donor"], [0, 0])
        self.assertEqual(row["conflicts"], 1)
        self.assertEqual(row["target_only"], 1)

    def test_two_agreeing_systematic_observations_and_held_parity_rebuild_missing_bits(self):
        matrix = np.array([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=np.uint8)
        expected = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)
        received = np.tile(expected, (3, 1))
        received[0, 0] ^= 1
        received[1, 1] ^= 1
        received[2, 1] ^= 1
        valid = np.ones(received.shape, dtype=bool)
        valid[1, 1] = False
        fit = np.array([True, False, True, False])
        result, _, reconstructed = audit_code5_triplet(received, valid, matrix, [fit, ~fit])
        self.assertTrue(result["erasure_validation"]["validated"])
        np.testing.assert_array_equal(reconstructed, expected)
