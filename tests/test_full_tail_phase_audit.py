import unittest

import numpy as np

import project_paths
from audit_full_tail_phase import classify_tail, endpoint_models, matched_tone_tail


class FullTailPhaseTests(unittest.TestCase):
    def test_tail_phase_and_constant_separate_on_held_period(self):
        rng = np.random.default_rng(2026112167)
        word = rng.choice([-1, 1], 60)
        count, start = 1860, 1473
        values = rng.choice([-1, 1], count).astype(complex)
        values[start:] = (1+2j)*word[(np.arange(start, count)-start) % 60]
        tail = classify_tail(values, word)
        self.assertEqual(tail['kind'], 'periodic')
        self.assertEqual(tail['periodic_phase'], start % 60)
        self.assertEqual(tail['refined_start'], start)
        values[-60:] = (1+2j)*word[(np.arange(count-60, count)-start-1) % 60]
        self.assertIsNone(classify_tail(values, word)['kind'])
        values[start:] = 1+2j
        values[:start] = 0
        constant = classify_tail(values, word)
        self.assertEqual(constant['kind'], 'constant')
        self.assertEqual(constant['coarse_start'], start)
        self.assertIsNone(constant['refined_start'])

    def test_word_and_allocation_rounding_are_distinct(self):
        descriptors = [{'mcs_candidate': 35, 'quantity': 3}]
        a, b = endpoint_models(228, 3, descriptors, {35: 5462})
        self.assertEqual(a-b, 2)

    def test_matched_tone_constant_allows_frequency_dependent_gain(self):
        rng = np.random.default_rng(2026112167)
        word = rng.choice([-1, 1], 60)
        common = np.exp(1j*rng.uniform(-np.pi, np.pi, 1004))
        same = np.tile(common, (4, 1))
        estimated = matched_tone_tail(same, word)
        self.assertEqual(estimated['kind'], 'constant')
        self.assertAlmostEqual(estimated['constant_held_skill'], 1.)
        resources = np.arange(301200-4016, 301200).reshape(4, 1004)
        cyclic = same*word[(resources-23) % 60]
        result = matched_tone_tail(cyclic, word)
        self.assertEqual(result['kind'], 'periodic')
        self.assertEqual(result['periodic_phase'], 23)
