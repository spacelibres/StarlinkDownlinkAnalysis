import unittest

import numpy as np

import project_paths
from audit_leo_full_code_observability import column_basis, disjoint_bases, observed_bits, residual_dimension


class LeoFullCodeObservabilityTests(unittest.TestCase):
    def test_upper_and_lower_resource_indices_wrap_and_preserve_last_lane(self):
        mask = np.array([True, False, False])
        self.assertEqual(np.flatnonzero(observed_bits(mask, 2, 2, 9)).tolist(), [2, 3, 8])

    def test_hidden_duplicate_bases_remain_full_rank_after_single_deletion(self):
        columns = [1, 1, 2, 2, 3]
        hidden = np.ones(len(columns), dtype=bool)
        first, second = disjoint_bases(columns, hidden, 2)
        self.assertEqual((len(first), len(second)), (2, 2))
        self.assertTrue(set(first).isdisjoint(second))
        for removed in range(len(columns)):
            self.assertEqual(len(column_basis(columns, (i for i in range(len(columns)) if i != removed), 2)), 2)

    def test_rank_accounts_for_optional_known_systematic_bits(self):
        columns = [1, 1, 2, 3]
        self.assertEqual(residual_dimension(columns, np.array([False, True, False, False]), 2), (1, 2))
        self.assertEqual(residual_dimension(columns, np.array([False, True, True, False]), 2), (0, 2))


if __name__ == "__main__":
    unittest.main()
