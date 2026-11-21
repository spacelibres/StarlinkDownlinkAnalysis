import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import project_paths
sys.path.insert(0, str(ROOT / 'experiments/E018/scripts'))
from qc_figure_core import dense_checks, qc_syndrome, generator_from_shifts, symbol_llrs, one_defect_span
from code21_ring import binary_blocks, evaluate, pack_blocks


class FigureTransferTests(unittest.TestCase):
    def test_shift_sign_and_systematic_inversion(self):
        shifts = np.array([[1, 2]])
        u = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        p = generator_from_shifts(shifts, 4)
        v = binary_blocks(evaluate(pack_blocks(u, 4), p, 4), 4)
        np.testing.assert_array_equal(v, np.roll(u, 1, axis=1))
        word = np.c_[u, v]
        self.assertFalse(qc_syndrome(word, shifts, 4).any())
        np.testing.assert_array_equal((word @ dense_checks(shifts, 4).T) % 2, qc_syndrome(word, shifts, 4))

    def test_symbol_decisions_preserve_serial_bit_order(self):
        points = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
        mapping = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        l = symbol_llrs(points, points, mapping)
        np.testing.assert_array_equal((l < 0).reshape(4, 2), mapping)

    def test_span_counts_nonunit_tail_using_a_second_column_pivot(self):
        self.assertEqual(one_defect_span([[0, 1], [3, 0]], 4),
                         {'unit_pivots': 1, 'residual_valuation': 1, 'binary_rank': 7})
        self.assertEqual(one_defect_span([[5, 0], [0, 1]], 4)['binary_rank'], 6)


if __name__ == '__main__':
    unittest.main()
