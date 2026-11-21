import unittest

import numpy as np
import project_paths
from audit_mcs85_record_boundary import prefix_sites, trained_boundary


class Mcs85RecordBoundaryTests(unittest.TestCase):
    def test_prefix_gate_uses_only_first_four_bytes_and_erases_incomplete_prefix(self):
        pattern = np.array([0x10, 0x20, 0xE8, 0xC0], dtype=np.uint8)
        stream = np.tile(np.r_[pattern, np.full(28, 0xFF, dtype=np.uint8)], 2)
        seen = np.full_like(stream, 0xFF)
        seen[33] = 0xFE
        self.assertEqual(prefix_sites(stream, seen, pattern).tolist(), [0])

    def test_boundary_uses_continuous_stable_and_variable_regions(self):
        expected = np.full(64, 0.55)
        expected[8:20] = 0.97
        self.assertEqual(trained_boundary(expected), 20)
