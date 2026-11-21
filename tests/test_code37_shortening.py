import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/E017/scripts'))
from code37_core import core_matrix, search_moves


class Code37ShorteningTests(unittest.TestCase):
    def test_core_matrix_ignores_unexplained_and_spare_positions(self):
        p = np.ones((5120, 3072), dtype=np.uint8)
        h = core_matrix(p)
        self.assertEqual(h.shape, (8193, 3072))
        self.assertTrue(h[:4736].all())
        self.assertFalse(h[4736:5120].any())
        self.assertFalse(h[-1].any())
        np.testing.assert_array_equal(h[5120:8192], np.eye(3072, dtype=np.uint8))

    def test_search_excludes_same_symbol_pairs_and_preserves_multiple_solutions(self):
        # Move layout is (symbol, new label, cost, syndrome delta).
        moves = [(0, 1, 1., 1), (0, 2, 2., 2), (1, 1, 3., 2), (2, 1, 4., 3)]
        candidates = search_moves(3, moves, pair_limit=3)
        self.assertEqual(candidates, [((2, 1),), ((0, 1), (1, 1))])
        self.assertEqual(search_moves(3, moves, pair_limit=2), [((2, 1),)])

    def test_four_move_search_joins_complementary_disjoint_pairs(self):
        moves = [(0, 1, 0., 1), (1, 1, 1., 2), (2, 1, 2., 4), (3, 1, 3., 8)]
        self.assertEqual(search_moves(15, moves, pair_limit=4, max_changes=4),
                         [((0, 1), (1, 1), (2, 1), (3, 1))])
        self.assertEqual(search_moves(7, moves, pair_limit=4, max_changes=4), [((0, 1), (1, 1), (2, 1))])


if __name__ == '__main__':
    unittest.main()
