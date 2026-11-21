from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import project_paths
sys.path.insert(0, str(ROOT / 'experiments/E019/scripts'))
from code28_core import coordinates, packed_check_columns
from code21_ring import dense_generator
from audit_code28_boundaries import last_symbol_indices, fit_pair


class Code28CoreTests(unittest.TestCase):
    def test_codeword_coordinates_do_not_consume_unknown_middle_or_spare(self):
        x = np.ones((2, 16386), dtype=np.uint8)
        u, v, mask = coordinates(x)
        self.assertEqual(u.shape, (2, 9216))
        self.assertTrue(np.all(u[:, :256] == 0))
        self.assertTrue(np.all(u[:, 256:] == 1))
        self.assertEqual(v.shape, (2, 7168))
        self.assertEqual(mask.sum(), 16128)
        self.assertFalse(mask[8960:9216].any())
        self.assertFalse(mask[-2:].any())

    def test_packed_check_columns_match_dense_convention(self):
        p = [[5, 3], [8, 2]]
        columns = packed_check_columns(p, 4, omitted=2, spare=1)
        expected = dense_generator(p, 4)
        integers = [sum(int(b) << i for i, b in enumerate(row)) for row in expected]
        self.assertEqual(columns, integers[2:] + [0, 0] + [1 << i for i in range(8)] + [0])

    def test_continuous_packing_carries_unused_symbol_capacity(self):
        np.testing.assert_array_equal(last_symbol_indices(4, None), [5461, 10922, 16383, 21845])
        np.testing.assert_array_equal(last_symbol_indices(4, 5462), [5461, 10923, 16385, 21847])
        self.assertEqual(fit_pair(np.array([2, 3, 1, 1, 3, 3])), [1, 3])


if __name__ == '__main__':
    unittest.main()
