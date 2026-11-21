import unittest

import numpy as np

import project_paths
from audit_leo_profiled_word import flip_descent, signed_design, best_phases


class LeoProfiledWordTests(unittest.TestCase):
    def test_signed_design_accumulates_overlapping_carrier_support(self):
        rng = np.random.default_rng(16)
        features = rng.normal(size=(13, 3, 15)) + 1j * rng.normal(size=(13, 3, 15))
        gains = np.array([2 - 1j, 1j, -.5 + 2j])
        indices = (np.array([1, 5, 3])[:, None] + 4 * np.arange(15)) % 60
        word = rng.choice([-1, 1], 60)
        expected = np.einsum('nki,ki,k->n', features, word[indices], gains)
        np.testing.assert_allclose(signed_design(features, indices, gains) @ word, expected)

    def test_flip_descent_minimizes_given_quadratic_and_preserves_unobserved(self):
        q = np.diag([3., 2., 0., 1.])
        h = np.array([-3., 2., 0., -1.])
        result, drops = flip_descent(q, h, np.ones(4, dtype=int))
        self.assertEqual(result.tolist(), [-1, 1, 1, -1])
        self.assertEqual(drops, [12., 4.])

    def test_equivalent_donor_phases_stay_erased(self):
        scores = np.array([[.1, .2, .2, .15], [.2, .1, .1, .2]])
        matches = best_phases(scores)
        self.assertEqual([row.tolist() for row in matches], [[1, 2], [0, 3]])
