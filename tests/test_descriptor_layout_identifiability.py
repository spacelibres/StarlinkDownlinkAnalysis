from itertools import product
import unittest

import numpy as np

import project_paths
from audit_descriptor_layout import modular_system, predict_residues


class DescriptorLayoutTests(unittest.TestCase):
    def test_modulo_four_solution_space_includes_nonunit_pivots(self):
        a = np.array([[2, 0, 2], [1, 2, 0], [0, 2, 2]])
        b = np.array([2, 3, 2])
        result = modular_system(a, b, 4)
        p, vectors = result['particular'], result['homogeneous']
        actual = {tuple((p+np.array(c)@vectors) % 4) for c in product(range(4), repeat=len(vectors))}
        brute = {x for x in product(range(4), repeat=3) if np.array_equal(a@x % 4, b)}
        self.assertEqual(actual, brute)

    def test_certificate_replays_modulo_four_and_modulo_five(self):
        for a, b, modulus in (([[2], [2]], [0, 2], 4), ([[2]], [1], 4), ([[1], [1]], [2, 3], 5)):
            a, b = np.array(a), np.array(b)
            witness = modular_system(a, b, modulus)['witness']
            c, m = witness['coefficients'], witness['modulus']
            self.assertTrue(np.all(c@a % m == 0))
            self.assertNotEqual(c@b % m, 0)

    def test_prediction_keeps_unobserved_units_as_erasures(self):
        a, b = np.array([[2, 0], [1, 0]]), np.array([86, 43])
        fits = [modular_system(a, b, m) for m in (3, 4, 5)]
        predicted, known = predict_residues(fits, np.array([[1, 0], [0, 1], [4, 60]]))
        np.testing.assert_array_equal(known, [True, False, True])
        np.testing.assert_array_equal(predicted[known], [43, 52])
