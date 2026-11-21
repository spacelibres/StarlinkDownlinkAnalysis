import unittest

import numpy as np

import project_paths
from audit_code5_to_code21_edge import learn_phase, recover_systematic_erasures


class EdgeTransferTests(unittest.TestCase):
    def test_phase_uses_reference_only_and_can_erase(self):
        g = .8*np.exp(1.3j)
        fit = learn_phase(np.repeat(g, 5))
        self.assertEqual(fit['quarter_turn'], 3)
        self.assertAlmostEqual(fit['correction_angle'], -1.3)
        self.assertIsNone(learn_phase(np.array([g, -g, g, -g])))

    def test_requires_observed_held_parity_and_keeps_unobserved_separate(self):
        gen = np.array([[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 0, 1]], np.uint8)
        residual = gen[0].copy()
        residual[-1] ^= 1  # Corruption in an erased parity coordinate.
        visible = np.ones(6, dtype=bool)
        visible[-1] = False
        masks = [np.array([True, True, False, False, False, False]),
                 np.array([False, False, True, True, False, False]),
                 np.array([False, False, False, False, True, True])]
        ok = recover_systematic_erasures(gen[:1], residual, visible, masks)
        self.assertTrue(ok['validated'])
        self.assertEqual(ok['systematic_guess'], [1])
        self.assertEqual(ok['unobserved_parity_count'], 1)
        no_coverage = visible.copy()
        no_coverage[4] = False
        self.assertFalse(recover_systematic_erasures(gen[:1], residual, no_coverage, masks)['validated'])
