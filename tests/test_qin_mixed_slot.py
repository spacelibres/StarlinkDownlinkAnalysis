import unittest

import numpy as np

import project_paths
from audit_qin_mixed_slot import ALPHABETS, hard_rings, split_evidence, modulation_delta


class MixedSlotTests(unittest.TestCase):
    def test_hard_rings_keep_shared_outer_radius(self):
        points = ALPHABETS['16QAM']
        hard = np.tile(points/max(abs(points)), 13)
        r = hard_rings(hard)
        self.assertEqual(r['16QAM_only']+r['shared'], len(hard))
        self.assertEqual(r['cross32_only'], 0)
        self.assertGreater(r['shared'], 0)

    def test_noiseless_mixed_observations_find_the_transition(self):
        rng = np.random.default_rng(61)
        n = 416
        left = rng.choice(ALPHABETS['16QAM'], n)
        right = rng.choice(ALPHABETS['cross32'], 1004-n)
        change = split_evidence(modulation_delta(np.r_[left, right], '16QAM', 'cross32', 1., 1., .03), n)
        self.assertEqual(change['best_resource_within_slot'], n)
        self.assertGreater(change['at_predicted_gain_over_homogeneous'], 0.)
        self.assertGreater(change['preboundary_mean'], 0.)
        self.assertLess(change['postboundary_mean'], 0.)
