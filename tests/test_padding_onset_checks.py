from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E083/scripts"))
from extract_mcs66_padding_sweep import measure_tail
from recover_mcs66_systematic_checks import cyclic_response, project_checks
from audit_mcs66_single_erasures import single_erasure_estimates


class PaddingOnsetCheckTests(unittest.TestCase):
    def test_conservative_bound_excludes_erasures_and_rounds_up(self):
        pn = np.resize([0, 1, 0, 0, 1, 1, 0], 2048).astype(np.uint8)
        bits = pn.copy()
        bits[:131] ^= 1
        bits[170] ^= 1
        seen = np.ones_like(pn, dtype=bool)
        seen[170] = False
        result = measure_tail(bits, seen, pn, 1900)
        self.assertEqual(result['onset'], 131)
        self.assertEqual(result['bound'], 136)
        self.assertEqual(result['pn_errors'], 0)
        self.assertEqual(result['pn_seen'], 1900 - 136 - 1)

    def test_cyclic_response_matches_scalar_rotation(self):
        active = np.array([[0, 1, 1, 0], [1, 1, 0, 1]], dtype=bool)
        weights = np.array([[1.0, 2.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.5]])
        expected = [np.sum(np.roll(active, -i, axis=1) * weights) for i in range(4)]
        np.testing.assert_allclose(cyclic_response(active, weights), expected, atol=1e-8)

    def test_zero_check_requires_all_involved_positions_observed(self):
        bits = np.array([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=bool)
        mask = np.array([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
        syndrome, seen = project_checks(bits, mask, np.array([[0, 1], [1, 3]]))
        np.testing.assert_array_equal(syndrome, [[True, False], [False, True]])
        np.testing.assert_array_equal(seen, [[True, False], [True, True]])

    def test_single_erasure_estimates_omit_conflicting_checks(self):
        support = np.array([[0, 1, 2], [0, 3, 4], [5, 2, 4], [5, 1, 6]])
        bits = np.array([0, 1, 0, 0, 0, 0, 1], dtype=np.uint8)
        mask = np.array([False, True, True, True, True, False, True])
        estimated, support_counts, conflict = single_erasure_estimates(bits, mask, support)
        self.assertEqual(estimated, {5: 0})
        self.assertEqual(support_counts, {5: 2})
        self.assertEqual(conflict, [0])
