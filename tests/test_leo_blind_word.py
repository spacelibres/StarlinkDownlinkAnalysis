import unittest

import numpy as np

import project_paths
from audit_leo_blind_word import blind_word, consensus_positions, cyclic_distance, profile_background


class LeoBlindWordTests(unittest.TestCase):
    def test_blind_multiframe_recovers_word_up_to_lane_polarity(self):
        rng = np.random.default_rng(2026112166)
        word = rng.choice([-1, 1], 60)
        q = np.array([16, 17, 13, 14, 15])
        phases = np.array([12, 23, 47, 58, 2, 53, 27])
        index = (phases[:, None, None]+q[None, :, None]+1004*np.arange(15)) % 60
        values = word[index].astype(complex)
        values -= values.mean(axis=-1, keepdims=True)
        phase = np.exp(1j*rng.uniform(-np.pi, np.pi, values.shape[:2]))
        values *= phase[..., None]
        estimate = blind_word(values, q)
        flipped = word*np.tile([1, -1, 1, -1], 15)
        self.assertEqual(cyclic_distance(estimate['word'], word)['mismatches'], 0)
        self.assertEqual(cyclic_distance(estimate['word'], flipped)['mismatches'], 0)

    def test_held_values_do_not_change_training_coefficients(self):
        rng = np.random.default_rng(166)
        design = rng.normal(size=(93, 3))
        other = rng.normal(size=(93, 2, 15))+1j*rng.normal(size=(93, 2, 15))
        signal = rng.normal(size=93).astype(complex)
        cal = np.arange(93) % 5 != 0
        initial = profile_background(design, other, signal, cal)
        signal[~cal] *= -500j
        altered = profile_background(design, other, signal, cal)
        np.testing.assert_array_equal(initial['calibration_coefficients'], altered['calibration_coefficients'])

    def test_disagreement_remains_erased_after_alignment(self):
        rng = np.random.default_rng(2026112166)
        word = rng.choice([-1, 1], 60)
        changed = word.copy()
        changed[[16, 50]] *= -1
        shifted = np.roll(changed*np.tile([-1, -1, 1, 1], 15), 28)
        result = consensus_positions([word, shifted])
        self.assertEqual(result['erased_positions'], [16, 50])
        self.assertEqual(result['canonical_signs'][16], 0)
        self.assertEqual(result['canonical_signs'][50], 0)
