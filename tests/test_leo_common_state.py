import unittest

import numpy as np

import project_paths
from audit_leo_common_state import fit_state, target_coordinates, transfer_scores


class LeoCommonStateTests(unittest.TestCase):
    def test_word_positions_map_to_different_slots_on_different_carriers(self):
        positions = np.array([42, 26, 34])
        np.testing.assert_array_equal(target_coordinates([537, 536], 1, positions),
                                      [(0, 6), (0, 7), (0, 14)])
        np.testing.assert_array_equal(target_coordinates([525, 526, 527], 1, positions),
                                      [(0, 2), (0, 3), (0, 10)])
        self.assertIsNone(target_coordinates([536, 537], 0, positions))

    def test_calibration_recovers_state_and_never_reads_held_samples(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(90, 2, 15)) + 1j*rng.normal(size=(90, 2, 15))
        word = np.where(np.arange(60) % 3, 1, -1)
        q = np.array([16, 17])
        indices = (q[:, None] + 1004*np.arange(15) + 1) % 60
        expected = word[indices].copy()
        expected[1, [6, 7, 14]] *= -1
        samples = np.einsum('nki,ki,k->n', features, expected, [1+1j, .4-.8j])
        training = np.arange(len(samples)) % 2 == 0
        data = {'features': features, 'target': samples.copy(), 'training': training}
        a = fit_state(data, [536, 537], 1, word, [42, 26, 34])
        data['target'][~training] *= -1
        b = fit_state(data, [536, 537], 1, word, [42, 26, 34])
        self.assertEqual(a['donor_sign'], -1)
        self.assertEqual(b['donor_sign'], -1)
        self.assertAlmostEqual(a['donor_projection'], -1.)
        self.assertAlmostEqual(a['held_projection'], -1.)
        self.assertNotAlmostEqual(a['held_projection'], b['held_projection'])

    def test_transfer_scores_compare_both_static_polarities_and_shift_frame_signs(self):
        result = transfer_scores([1, -1, 1, -1], [3., -3., 3., -3.], [2., 2., 2., 2.], [10.]*4)
        self.assertEqual(result['used_frames'], 4)
        self.assertAlmostEqual(result['observed_improvement'], .4)
        self.assertAlmostEqual(result['constant_positive_improvement'], -.2)
        self.assertAlmostEqual(result['constant_negative_improvement'], -.2)
        self.assertEqual(result['rolled_improvements_ge_observed'], 1)

    def test_profiled_constant_carrier_does_not_erase_other_carrier(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(96, 2, 15)) + 1j*rng.normal(size=(96, 2, 15))
        features -= features.mean(axis=2, keepdims=True)
        word = np.ones(60, dtype=int)
        word[[2, 26, 34, 58]] = -1
        indices = (np.array([16, 17])[:, None] + 1004*np.arange(15) + 1) % 60
        expected = word[indices].copy()
        expected[1, [6, 7, 14]] *= -1
        samples = np.einsum('nki,ki,k->n', features, expected, [1+1j, .4-.8j])
        training = np.arange(len(samples)) % 2 == 0
        data = {'features': features, 'target': samples, 'training': training}
        result = fit_state(data, [536, 537], 1, word, [42, 26, 34])
        self.assertTrue(result['qualified'])
        self.assertAlmostEqual(result['held_projection'], -1.)
        erased = fit_state(data, [536, 537], 1, word, [41, 25, 33])
        self.assertFalse(erased['qualified'])
