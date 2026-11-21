import unittest

import numpy as np

import project_paths
from audit_leo_cfo_branches import aperture_scores, score_splits


class LeoCfoBranchTests(unittest.TestCase):
    def test_full_symbol_samples_distinguish_aliases(self):
        rng = np.random.default_rng(5601)
        slot_s = 4.4e-6
        local = np.arange(1, 10) * .4e-6
        time = (np.arange(300)[:, None] * slot_s + local).ravel()
        early = np.tile(local < slot_s/2, 300)
        frequencies = (np.arange(8)-3.5)*234375
        states = np.exp(.5j*np.pi*rng.integers(0, 4, size=(300, 8)))
        basis = np.repeat(states, len(local), axis=0) * np.exp(
            2j*np.pi*np.tile(local, 300)[:, None]*frequencies
        )
        offsets = np.array([-1., 0., 1.])/slot_s
        gains = rng.normal(size=8)+1j*rng.normal(size=8)
        y = (basis @ gains)*np.exp(2j*np.pi*offsets[2]*time)
        energies, total = aperture_scores(basis, y, time, early, offsets)
        self.assertAlmostEqual(energies[2]/total, 1.)
        self.assertLess(max(energies[:2]/total), .1)

    def test_frequency_selection_uses_only_calibration_group(self):
        improvements = np.zeros((14, 2, 2))
        improvements[:, 0, 0] = 1
        improvements[7:, 0, 1] = 8
        rows = score_splits(improvements, np.ones(14), ['reference', 'alias'])
        self.assertEqual(len(rows), 6)
        first = rows[0]
        self.assertEqual(first['held_frames'], list(range(7, 14)))
        self.assertEqual(first['models']['reference']['grid_index'], 0)
        self.assertEqual(first['models']['reference']['held_improvement'], 1)
