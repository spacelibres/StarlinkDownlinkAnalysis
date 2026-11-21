import unittest

import numpy as np

import project_paths
from extract_mcs70_terminal_words import selected_terminal_words
from recover_mcs70_sparse_families import nominate_shapes
from compare_weak_mcs66_mcs70 import pn_likelihood_ratio
from transfer_mcs_checks_to_weak import extrinsic_bits


class Mcs70CheckSearchTests(unittest.TestCase):
    def test_preserves_stop_confirmation_and_terminal_word_index(self):
        a = {'label': 70, 'count': 7, 'start': 40, 'width_resources': 4096}
        b = {'label': 66, 'count': 4, 'start': 90, 'width_resources': 4096}
        obs = [{'label': 70, 'start': 40, 'masked_transfer_stop_words': [0, 6]},
               {'label': 66, 'start': 90, 'masked_transfer_stop_words': [3]}]
        self.assertEqual(selected_terminal_words(obs, [a, b]), [(a, 6)])
        obs[0]['masked_transfer_stop_words'] = [0]
        self.assertEqual(selected_terminal_words(obs, [a, b]), [])

    def test_packed_difference_signatures_recover_four_block_relation(self):
        rng = np.random.default_rng(662)
        observed = rng.integers(0, 2, size=(24, 32)).astype(bool)
        observed[:, 29] = observed[:, 0] ^ observed[:, 11] ^ observed[:, 22]
        mask = np.ones_like(observed, bool)
        shape = (0, 11, 22, 29)
        self.assertIn(shape, nominate_shapes(observed, mask, 8))
        mask[0, 29] = False
        self.assertNotIn(shape, nominate_shapes(observed, mask, 8))

    def test_symbol_mixture_channel_ratio_prefers_known_labels(self):
        points = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
        intended = points[[0, 2, 1, 3]]
        channel = np.array([1., .7j, 2., -1j])
        data = channel*intended
        right = pn_likelihood_ratio(data, channel, points, intended, .1)
        wrong = pn_likelihood_ratio(data, channel, points, np.roll(intended, 1), .1)
        self.assertTrue(np.all(right > 1.))
        self.assertTrue(np.all(wrong < 0.))

    def test_extrinsic_prediction_excludes_the_scored_bit(self):
        support = np.array([[0, 4, 9], [1, 6, 10]])
        moments = np.arange(24, dtype=float).reshape(6, 4)/24
        first, held = extrinsic_bits(moments, support)
        moments.ravel()[support[:, -1]] *= -1
        again, neg = extrinsic_bits(moments, support)
        np.testing.assert_array_equal(again, first)
        np.testing.assert_array_equal(neg, -held)
