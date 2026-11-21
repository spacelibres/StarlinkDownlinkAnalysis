import unittest

import numpy as np

import project_paths
from audit_leo_tcode_iq_prediction import score_families


class LeoTcodeIQTests(unittest.TestCase):
    def test_profiled_model_equals_explicit_complex_regression(self):
        rng = np.random.default_rng(2026112165)
        cal = np.arange(180) % 3 != 0
        nuisance = rng.normal(size=(180, 5)) + 1j*rng.normal(size=(180, 5))
        classes = np.arange(180) % 15
        atoms = rng.normal(size=(180, 2)) + 1j*rng.normal(size=(180, 2))
        expanded = atoms[:, :, None]*(classes[:, None, None] == np.arange(15))
        waves = rng.choice([-1., 1.], size=(4, 3, 2, 15))
        matching = np.einsum('nki,ki->nk', expanded, waves[0, 2])
        signal = nuisance @ np.arange(1, 6) + matching @ [1.5-2j, 4j]
        output = score_families(nuisance, expanded, signal, cal, waves)
        self.assertEqual(output['phase'][0], 2)
        self.assertAlmostEqual(output['held_error_reduction'][0], output['held_background_error'])
        self.assertAlmostEqual(output['reference_phase_held_error_reduction'][2], output['held_background_error'])
        self.assertLess(output['reference_phase_held_error_reduction'][1], output['held_background_error'])
        for family, phase in enumerate(output['phase']):
            candidates = np.einsum('nki,ki->nk', expanded, waves[family, phase])
            design = np.column_stack((nuisance, candidates))
            coefficient = np.linalg.lstsq(design[cal], signal[cal], rcond=None)[0]
            improve = output['held_background_error']-np.linalg.norm(signal[~cal]-design[~cal] @ coefficient)**2
            self.assertAlmostEqual(output['held_error_reduction'][family], improve)
        signal[~cal] *= -1
        swapped = score_families(nuisance, expanded, signal, cal, waves)
        np.testing.assert_array_equal(output['phase'], swapped['phase'])
