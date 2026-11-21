import unittest

import numpy as np

import project_paths
from audit_pn_padding_starts import grid_evidence, word_windows


class PnPaddingStartTests(unittest.TestCase):
    def test_integrated_evidence_is_normalized_for_constant_profile(self):
        values = np.ones(177)*97.
        evidence = grid_evidence(values)
        for row in evidence:
            np.testing.assert_allclose(row, 0., atol=1e-12)

    def test_coverage_and_transition_are_required(self):
        pn = np.tile([1, 0, 1, 0], 256).astype(np.uint8)
        bits = pn.copy()
        bits[512::2] ^= 1
        valid = np.ones(1024, dtype=bool)
        self.assertTrue(word_windows(bits[None], valid[None], pn, 512)['passed'][0])
        valid[:60] = False
        self.assertFalse(word_windows(bits[None], valid[None], pn, 512)['passed'][0])
        self.assertFalse(word_windows(pn[None], np.ones((1, 1024), bool), pn, 512)['passed'][0])
