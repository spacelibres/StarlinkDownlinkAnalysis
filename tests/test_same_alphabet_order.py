from itertools import permutations
import unittest

import numpy as np

import project_paths
from audit_same_alphabet_order import ordering, padding_marker


class SameAlphabetOrderTests(unittest.TestCase):
    def test_dynamic_program_tracks_all_optimal_orderings(self):
        rng = np.random.default_rng(2026112169)
        n = 5
        edges = rng.integers(0, 2, (1 << n, n))
        score_map = {}
        for p in permutations(range(n)):
            mask, score = 0, 0
            for i in p:
                score += edges[mask, i]
                mask |= 1 << i
            score_map[p] = score
        hi = max(score_map.values())
        best = [p for p, s in score_map.items() if s == hi]
        result = ordering(edges)
        self.assertEqual(result['best'], hi)
        self.assertEqual(result['best_count'], len(best))
        for label in range(n):
            self.assertEqual(result['possible_positions'][label], sorted({p.index(label) for p in best}))

    def test_marker_requires_a_stop_not_continuous_pn(self):
        before = np.zeros((128, 4), dtype=np.uint8)
        after = np.zeros((128, 4), dtype=np.uint8)
        post = after.copy()
        post[::2] = 1
        self.assertTrue(padding_marker(before, post, before, after)['passed'])
        self.assertFalse(padding_marker(before, after, before, after)['passed'])
