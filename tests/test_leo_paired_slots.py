import unittest

import numpy as np

import project_paths
from audit_leo_paired_slots import cycle_differences, partition_groups, transfer, subtract_pilots


class PairedSlotTests(unittest.TestCase):
    def test_cycle_difference_cancels_repetition_and_groups_do_not_overlap(self):
        periodic = np.tile(np.arange(15), 20)[None, :]
        np.testing.assert_array_equal(cycle_differences(periodic), np.zeros((1, 6, 2, 15)))
        train, held = partition_groups()
        self.assertEqual(len(train), 3)
        self.assertEqual(len(held), 3)
        self.assertEqual(set(train) | set(held), set(range(6)))
        self.assertFalse(set(train) & set(held))

    def test_true_pair_predicts_after_calibration_only_pilot_subtraction(self):
        rng = np.random.default_rng(2026112157)
        make = lambda shape: rng.normal(size=shape) + 1j*rng.normal(size=shape)
        truth = make((4, 6, 2, 15))
        design = make((6, 2, 15, 24))
        gains = make((4, 24))
        first = truth + np.einsum('gtsp,fp->fgts', design, gains)
        second = truth * np.exp(1j*np.arange(4))[:, None, None, None]
        train, held = partition_groups()
        first = subtract_pilots(first, design, train)
        second = subtract_pilots(second, design, train)
        error, coherence = transfer(first, second, train, held)
        wrong_error, _ = transfer(first, np.roll(second, 1, axis=0), train, held)
        self.assertAlmostEqual(coherence, 1.)
        self.assertGreater(error, .6)
        self.assertGreater(error, wrong_error+.5)

    def test_pilot_only_spill_is_removed_at_held_slots(self):
        rng = np.random.default_rng(2157)
        design = rng.normal(size=(6, 2, 15, 24))
        gains = rng.normal(size=(3, 24)) + 1j*rng.normal(size=(3, 24))
        values = np.einsum('gtsp,fp->fgts', design, gains)
        train, held = partition_groups()
        residual = subtract_pilots(values, design, train)
        np.testing.assert_allclose(residual[:, held], 0., atol=1e-12)
