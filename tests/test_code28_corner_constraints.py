import unittest

import numpy as np
import project_paths

from audit_code28_corner_constraints import solve_domains


class Code28CornerConstraintTests(unittest.TestCase):
    def test_coupled_change_uses_the_xor_of_two_syndrome_columns(self):
        columns = [1, 2, 3, 4]
        baseline = np.array([0, 0, 1, 0], dtype=np.uint8)
        repaired, rank, consistent = solve_domains(columns, baseline, [(0, 1), (3,)])
        self.assertTrue(consistent)
        self.assertEqual(rank, 2)
        np.testing.assert_array_equal(repaired, [1, 1, 1, 0])

    def test_an_out_of_span_syndrome_does_not_create_a_word(self):
        repaired, rank, consistent = solve_domains([1, 2, 3], np.array([0, 0, 1]), [(0,)])
        self.assertIsNone(repaired)
        self.assertEqual(rank, 1)
        self.assertFalse(consistent)
