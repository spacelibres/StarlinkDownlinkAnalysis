import unittest

import numpy as np

import project_paths
from audit_qam_pn15 import matching_run, ordered_indices


class QAMPN15Tests(unittest.TestCase):
    def test_byte_reversal_keeps_byte_number(self):
        self.assertEqual(ordered_indices(np.array([0, 3, 4, 7, 8, 15, 16]), 8).tolist(),
                         [7, 4, 3, 0, 15, 8, 23])

    def test_matching_run_preserves_exclusive_end_and_noise(self):
        difference = np.ones(500, dtype=bool)
        difference[100:400] = False
        difference[160] = True
        r = matching_run(difference)
        self.assertEqual((r["start"], r["stop"], r["mismatches"]), (100, 400, 1))


if __name__ == "__main__":
    unittest.main()
