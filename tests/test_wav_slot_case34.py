import json
import unittest

import numpy as np

import project_paths
from audit_wav_slot_case34 import compare_amplitudes, smooth_columns, summarize


class WavSlotCase34Tests(unittest.TestCase):
    def test_held_errors_distinguish_opposite_suppressed_and_reference(self):
        rng = np.random.default_rng(34)
        physical = rng.normal(size=(100, 4)) + 1j * rng.normal(size=(100, 4))
        target = rng.normal(size=100) + 1j * rng.normal(size=100)
        scales = np.ones_like(physical.real)
        gains = np.asarray([1j, 2 - 1j, 1 + .7j, .5j])
        fit = np.arange(100) % 2 == 0
        for coefficient in (-1, 0, 1):
            y = physical @ gains + coefficient * gains[2] * target
            result = compare_amplitudes(y, physical, target, 2, scales, 0, fit, ~fit)
            self.assertEqual(result['winner'], coefficient)
            self.assertAlmostEqual(result['free_coefficient'][0], coefficient)
            self.assertAlmostEqual(result['free_coefficient'][1], 0.0)

    def test_smooth_basis_is_carrier_major(self):
        self.assertEqual(smooth_columns(np.array([[2., 3.]]), np.array([[.5, -.5]]), 2).tolist(),
                         [[2., 1., .5, 3., -1.5, .75]])

    def test_summary_has_json_compatible_counts(self):
        target = {'held_nmse_reference_zero_flip': [1., .9, 1.], 'free_coefficient': [.12, .04], 'winner': 0}
        row = {'time_group': 0, 'arms': [{'alignment': {'qualified': True},
                                        'paths': [{'target': target, 'other_class_winners': {}}] * 3}]}
        self.assertEqual(json.loads(json.dumps(summarize([row])))[0]['free_inside_half_amplitude'], 1)
