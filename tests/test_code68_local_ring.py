import unittest

import project_paths
from audit_code68_qc_rank import solve_and_predict, truncated_product


class LocalRingTests(unittest.TestCase):
    def test_nonunit_pivot_and_modulo_prediction(self):
        #  tP = t+t² fixes P mod t³. It fixes tP mod t⁴.
        result = solve_and_predict([[2]], [[6]], [[1], [2]], [[3], [6]], 4)
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["prediction_fixed_prefixes"], [3, 4])
        self.assertEqual(result["held_mismatch_coefficients"], [0, 0])

    def test_reports_nonunit_contradiction(self):
        result = solve_and_predict([[4]], [[2]], [[1]], [[0]], 4)
        self.assertEqual(result["first_inconsistent_degree"], 1)

    def test_column_operations_preserve_the_image(self):
        # Select the right unit as first pivot and clear a nonzero column.
        p, q = 9, 3
        make = lambda row: truncated_product(row[0], p, 4) ^ truncated_product(row[1], q, 4)
        train, held = [[2, 1], [4, 2]], [[0, 1], [4, 0]]
        result = solve_and_predict(train, [[make(r)] for r in train], held, [[make(r)] for r in held], 4)
        self.assertEqual(result["rank"], 4)
        self.assertEqual(result["prediction_fixed_prefixes"], [1, 2])
        self.assertEqual(result["held_mismatch_coefficients"], [0, 0])
