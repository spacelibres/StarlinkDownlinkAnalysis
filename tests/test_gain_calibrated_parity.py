from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E082/scripts"))
from extract_gain_padded_words import conditioned_prefix
from audit_sparse_padding_checks import candidate_score, exact_candidates, shifted_scores
from audit_sparse_qc_orbits import canonical_support, orbit_positions


class GainCalibratedParityTests(unittest.TestCase):
    def test_padding_starts_at_rounded_conservative_bound(self):
        received = np.zeros(40, dtype=np.uint8)
        visible = np.zeros(40, dtype=bool)
        reference = np.ones(40, dtype=np.uint8)
        bits, mask, bound = conditioned_prefix(received, visible, reference, 9, 32)
        self.assertEqual(bound, 16)
        np.testing.assert_array_equal(bits[:16], 0)
        np.testing.assert_array_equal(bits[16:], 1)
        np.testing.assert_array_equal(mask[:16], False)
        np.testing.assert_array_equal(mask[16:], True)
        np.testing.assert_array_equal(received, 0)

    def test_sparse_search_finds_affine_pair_and_triple_with_unobserved_column_excluded(self):
        values = np.arange(32)[:, None]
        a, b, c, d, e = ((values[:, 0] >> i) & 1 for i in range(5))
        bits = np.column_stack((a, b, a ^ b ^ 1, c, c ^ 1, d, e, c))
        seen = np.ones_like(bits, dtype=bool)
        seen[0, 7] = False
        candidates, count = exact_candidates(bits, seen)
        self.assertEqual(count, 7)
        self.assertEqual(candidates, {((0, 1, 2), 1), ((3, 4), 1)})

    def test_batched_group_controls_match_scalar_scores(self):
        rng = np.random.default_rng(21)
        bits = rng.integers(0, 2, size=(20, 3), dtype=np.uint8)
        seen = rng.random(bits.shape) > .25
        group = rng.integers(0, 5, len(bits))
        origins = np.array([rng.permutation(len(bits)) for _ in range(7)])
        expected = [candidate_score(bits, seen, group, (0, 1, 2), 1, x)["group_accuracy"] for x in origins]
        self.assertEqual(shifted_scores(bits, seen, group, (0, 1, 2), 1, origins), expected)

    def test_block_rotation_wraps_within_each_block(self):
        support = ((0, 0), (1, 3), (1, 6))
        rows = orbit_positions(support, 8)
        self.assertEqual(rows[3].tolist(), [3, 14, 9])
        self.assertEqual(canonical_support(rows[3], 8), support)
