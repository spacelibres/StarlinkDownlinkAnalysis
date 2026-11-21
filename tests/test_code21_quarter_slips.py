import unittest

import numpy as np

import project_paths
from audit_code21_quarter_slips import (
    CHECK_BITS, WORD_SYMBOLS, apply_phase, phase_search, phase_deltas, syndrome_words,
    single_symbol_index, try_single_symbol,
)
from validate_code21 import coordinates


class Code21QuarterSlipTests(unittest.TestCase):
    def test_syndrome_deltas_compose_and_held_groups_reject_false_fit(self):
        rng = np.random.default_rng(62)
        generator = rng.integers(0, 2, (10, 6, 512), dtype=np.uint8)
        y = np.exp(1j*(np.pi/4+np.pi/2*rng.integers(0, 4, WORD_SYMBOLS)))
        slots = np.arange(WORD_SYMBOLS)//1004
        states = np.array([1, 2, 1, 3, 0])
        base, deltas = phase_deltas(y, slots, (1, 2), generator)
        actual_bits = coordinates(apply_phase(y, slots, states)[None, :], 1, 2)
        actual = syndrome_words(actual_bits, generator)[0]
        reconstructed = base ^ np.bitwise_xor.reduce(deltas[np.arange(len(states)), states], axis=0)
        np.testing.assert_array_equal(reconstructed, actual)

        desired_base = np.bitwise_xor.reduce(deltas[np.arange(len(states)), states], axis=0)
        result = phase_search(desired_base, deltas, uniform=False)
        self.assertTrue(result['validated'])
        self.assertEqual(result['accepted_states'], states.tolist())
        desired_base[CHECK_BITS-1] ^= 1
        self.assertFalse(phase_search(desired_base, deltas, uniform=False)['validated'])

    def test_single_symbol_requires_affected_held_checks(self):
        rng = np.random.default_rng(2162)
        p = rng.integers(0, 2, (16, CHECK_BITS), dtype=np.uint8)
        table = single_symbol_index(p)
        recovered = try_single_symbol(p[4] ^ p[5], table)
        self.assertTrue(recovered['validated'])
        self.assertEqual(recovered['selected_bit_positions'], [4, 5])
        parity_only = np.zeros(CHECK_BITS, dtype=np.uint8)
        parity_only[99] = 1
        result = try_single_symbol(parity_only, table)
        self.assertFalse(result['validated'])
        self.assertEqual(result['same_word_compatible_local_hypothesis'], [16+99])
