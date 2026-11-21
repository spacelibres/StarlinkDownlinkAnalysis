import unittest

import numpy as np

import project_paths
from replay_leo_padded_code import solve_erasures


class PaddedLeoErasureTests(unittest.TestCase):
    def test_exact_solver_recovers_unique_bit(self):
        solution, rank, consistent = solve_erasures([1, 2, 3], [True, False, True], [True, False, True])
        np.testing.assert_array_equal(solution, [True, True, True])
        self.assertEqual((rank, consistent), (1, True))

    def test_inconsistent_and_underdetermined_inputs_keep_erasures(self):
        self.assertEqual(solve_erasures([1, 2, 3], [True, False, False], [True, True, False]), (None, 1, False))
        self.assertEqual(solve_erasures([1, 1, 2], [False, False, False], [False, False, True]), (None, 1, True))
