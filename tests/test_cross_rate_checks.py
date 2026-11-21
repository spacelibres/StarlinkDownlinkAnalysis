import unittest

import numpy as np
import project_paths
from audit_cross_rate_checks import translate_rows


class CrossRateChecksTests(unittest.TestCase):
    def test_shift_systematic_only_and_preserve_parity_columns(self):
        rows = np.array([[514, 12017, 12807, 15000]])
        np.testing.assert_array_equal(translate_rows(rows, 12800, 12288),
                                      [[2, 11505, 12807, 15000]])
        self.assertIsNone(translate_rows(rows, 12800, 11776))
        self.assertIsNone(translate_rows(np.array([[4, 12292]]), 12288, 12800))
