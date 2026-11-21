import unittest

import numpy as np

import project_paths
from audit_wav_word_state import build_symbols, fit_bank, score_selected
from compare_wav_word_states import compare_patterns


class WavWordStateTests(unittest.TestCase):
    def test_local_and_rolling_word_are_different_hypotheses(self):
        resources = np.arange(60).reshape(4, 15)
        word = np.where(np.arange(60) % 7 < 3, 1, -1)
        symbols = build_symbols(word, resources, 2, [4, 5, 12])
        self.assertEqual(symbols.shape, (2, 60, 4, 15))
        np.testing.assert_array_equal(symbols[0, 8], word[(resources - 8) % 60])
        np.testing.assert_array_equal(symbols[0, 8, 2, [4, 5, 12]], -symbols[1, 8, 2, [4, 5, 12]])
        self.assertEqual(np.count_nonzero(symbols[0, 8] != symbols[1, 8]), 3)

    def test_candidate_uses_calibration_only_and_predicts_held_iq(self):
        rng = np.random.default_rng(84)
        design = rng.normal(size=(5, 30, 3)) + 1j*rng.normal(size=(5, 30, 3))
        mask = np.arange(30) % 2 == 0
        gain = np.array([1+.3j, -1.3j, 2-.4j])
        samples = np.stack([design[1] @ gain, design[3] @ gain])
        bank = fit_bank(design, mask)
        best = np.argmax(bank.objectives(samples), axis=0)
        np.testing.assert_array_equal(best, [1, 3])
        err = score_selected(bank, samples, best, ~mask)
        np.testing.assert_allclose(err, 0, atol=1e-24)
        samples[:, ~mask] *= -1
        np.testing.assert_array_equal(np.argmax(bank.objectives(samples), axis=0), best)
        self.assertTrue(np.all(score_selected(bank, samples, best, ~mask) > 1))

    def test_joint_pattern_comparison_keeps_prediction_erasures(self):
        observed = np.asarray([[1, 1, 1], [-1, -1, -1], [1, 0, -1], [1, 1, -1]])
        predicted = np.array([[[1, 1, 1], [-1, -1, -1], [0, 0, 0], [1, 1, -1]],
                              [[1, 1, 1], [1, 1, 1], [0, 0, 0], [-1, 1, -1]]])
        individual, joint = compare_patterns(observed, predicted)
        self.assertEqual(individual[0]['---'], {'---': 1})
        self.assertEqual(joint, {'+++': {'+++': 1}, '++-': {'erased': 1}, '---': {'erased': 1}})
