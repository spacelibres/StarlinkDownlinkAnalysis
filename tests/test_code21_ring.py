import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/E016/scripts'))
from code21_ring import binary_blocks, dense_generator, evaluate, pack_blocks, recover, ring_multiply, single_bit_candidates


class Code21RingTests(unittest.TestCase):
    def test_cyclic_multiplication_wraps_and_cancels(self):
        self.assertEqual(ring_multiply(0b1000, 0b0010, 4), 1)
        self.assertEqual(ring_multiply(0b1111, 0b0011, 4), 0)

    def test_ring_solver_matches_binary_generator_and_rejects_errors(self):
        inputs = [[0b1110, 1], [1, 0b1010], [0b1111, 0b1100]]
        expected = [[5, 3], [8, 2]]
        parity = evaluate(inputs, expected, 4)
        recovered, pivots, inconsistent = recover(inputs, parity, 4)
        self.assertEqual((recovered, pivots, inconsistent), (expected, 2, 0))
        dense = dense_generator(expected, 4)
        bits = binary_blocks(inputs, 4)
        np.testing.assert_array_equal((bits @ dense) % 2, binary_blocks(parity, 4))
        self.assertEqual(pack_blocks(bits, 4), inputs)
        parity[-1][-1] ^= 1
        self.assertEqual(recover(inputs, parity, 4), (None, 2, 1))

    def test_rank_deficient_residue_does_not_claim_unique_recovery(self):
        self.assertEqual(recover([[3]], [[3]], 4), (None, 0, None))

    def test_single_bit_syndrome_keeps_ambiguous_and_missing_candidates(self):
        p = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        self.assertEqual(single_bit_candidates(p, np.array([1, 1, 0])), [0])
        self.assertEqual(single_bit_candidates(p, np.array([1, 0, 0])), [1, 2])
        self.assertEqual(single_bit_candidates(p, np.array([0, 1, 1])), [])
        self.assertEqual(single_bit_candidates(p, np.array([0, 0, 0])), [])


if __name__ == '__main__':
    unittest.main()
