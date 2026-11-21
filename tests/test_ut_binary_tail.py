import unittest

import numpy as np

import project_paths
from audit_ut_binary_tail import binary_tail, constellation_costs, minimum_period, score_template


class BinaryTailTests(unittest.TestCase):
    def test_boundary_is_first_binary_resource(self):
        delta = np.r_[np.full(300, 0.4), np.full(83, -0.6)]
        self.assertEqual(binary_tail(delta, min_side=32)["first_binary_resource"], 300)

    def test_pure_region_is_not_a_transition(self):
        for score in (0.4, -0.6):
            self.assertIsNone(binary_tail(np.full(383, score), min_side=32)["first_binary_resource"])

    def test_fixed_costs_separate_binary_and_square(self):
        z = np.array([1, -1, (3+3j)/np.sqrt(10), (-1+3j)/np.sqrt(10)])
        a, b = constellation_costs(z, 0.03)
        np.testing.assert_array_equal(a < b, [True, True, False, False])

    def test_late_discrepancies_cannot_select_edge_shift(self):
        template = np.array([1, 1, 0, 0, 1, 0, 0], dtype=bool)
        decisions = template[(np.arange(320) - 1) % len(template)]
        decisions[130:] = ~decisions[130:]
        symbols = 1 - 2 * decisions.astype(float)
        self.assertEqual(minimum_period([np.tile(template, 5)]), 7)
        score = score_template(template, symbols, 290000)
        self.assertEqual(score["edge_adjustment"], -1)
        self.assertEqual(score["edge_fit_disagreements"], 0)
        self.assertEqual(score["held_disagreements"], 190)


if __name__ == "__main__":
    unittest.main()
