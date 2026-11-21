import unittest

import numpy as np

import project_paths
from audit_qc_soft_folds import qam_parity_moments, search_masks, fit_projection


class SoftFoldTests(unittest.TestCase):
    def test_qam_marginal_keeps_bits_within_a_symbol_together(self):
        y = np.array([[0.]])
        points = np.array([-1., 1.])
        labels = np.array([[0, 0], [1, 1]])
        # Both individual bits are completely ambiguous but their parity is even.
        np.testing.assert_allclose(qam_parity_moments(y, points, labels, .5), 1.)

    def test_affine_projection_recovers_two_training_columns(self):
        rng = np.random.default_rng(60)
        u = rng.choice([-1., 1.], size=(65, 7))*.93
        v = (-u[:, 1] * u[:, 5])[:, None]
        train = np.arange(65) < 31
        masks = search_masks(7, 3)
        rows, prediction = fit_projection(u, v, train, masks)
        self.assertEqual(rows[0]['system_columns'], [1, 5])
        self.assertEqual(rows[0]['sign'], -1)
        self.assertTrue(np.all(np.sign(prediction[~train, 0]) == np.sign(v[~train, 0])))
