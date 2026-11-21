from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E084/scripts"))
from search_mcs66_early_projections import signed_pair_index, matching_pairs, reduce_support, insert_support
from extend_mcs66_parity_matrix import exact_completions
from audit_mcs66_rotation_differences import paired_differences


class ExpandedMcs66Tests(unittest.TestCase):
    def test_pair_index_finds_a_match_and_its_affine_complement(self):
        signatures = np.array([3, 9, 17, 48], dtype=np.uint64)
        pairs, keys = signed_pair_index(signatures, np.arange(4), np.uint64(63))
        target = signatures[0] ^ signatures[2]
        self.assertEqual(list(matching_pairs(pairs, keys, np.array([target, target ^ np.uint64(63)]), np.uint64(63))),
                         [(0, 0, 2), (1, 0, 2)])

    def test_known_combinations_do_not_increase_rank(self):
        basis = {}
        self.assertTrue(insert_support((1, 3, 6), basis))
        self.assertTrue(insert_support((2, 3, 7), basis))
        self.assertEqual(reduce_support((1, 2, 6, 7), basis), 0)
        self.assertFalse(insert_support((1, 2, 6, 7), basis))
        self.assertTrue(insert_support((0, 2, 7), basis))

    def test_completion_keeps_exact_aliases_for_calibration_to_disambiguate(self):
        dictionary = np.array([[1, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
        self.assertEqual(exact_completions(dictionary, dictionary[0]), [(0,), (1, 2)])

    def test_differences_remove_common_bits_and_preserve_erasures(self):
        bits = np.array([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=bool)
        mask = np.ones_like(bits)
        mask[0, 2] = False
        delta, seen = paired_differences(bits, mask, 0, np.array([1]), np.array([1]), 2)
        np.testing.assert_array_equal(delta, [[False, True, False, False]])
        np.testing.assert_array_equal(seen, [[True, True, True, False]])
