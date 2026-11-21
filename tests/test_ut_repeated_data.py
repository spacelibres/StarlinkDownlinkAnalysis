import unittest

import numpy as np

import project_paths
from audit_ut_repeated_data import fundamental_period, learn_pattern, scan_pattern


class RepeatedDataTests(unittest.TestCase):
    def test_fundamental_period_may_not_be_byte_aligned(self):
        motif = np.unpackbits(np.array([0xb8, 0x80, 0x0b], dtype=np.uint8), bitorder="little")
        self.assertEqual(fundamental_period(motif), 12)

    def test_pattern_learning_preserves_absolute_word_phase(self):
        rng = np.random.default_rng(717)
        sample = rng.integers(0, 2, 512, dtype=np.uint8)
        motif = np.unpackbits(np.array([0xb8, 0x80, 0x0b], dtype=np.uint8), bitorder="little")
        sample[32:272] = motif[np.arange(32, 272) % len(motif)]
        model, _ = learn_pattern(sample[None, :], [7])
        np.testing.assert_array_equal(model, motif)

    def test_frozen_pattern_distinguishes_zero_fill_and_random_bits(self):
        rng = np.random.default_rng(897)
        sample = rng.integers(0, 2, 512, dtype=np.uint8)
        motif = np.unpackbits(np.array([0xb8, 0x80, 0x0b], dtype=np.uint8), bitorder="little")
        sample[39:201] = motif[(np.arange(39, 201) + 16) % len(motif)]
        fit = scan_pattern(sample, motif)
        self.assertTrue(fit["strict"])
        self.assertEqual(fit["byte_phase"], 2)
        self.assertFalse(scan_pattern(np.zeros(512, dtype=np.uint8), motif)["strict"])


if __name__ == "__main__":
    unittest.main()
