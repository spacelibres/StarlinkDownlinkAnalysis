import unittest

import numpy as np

import project_paths
from audit_ut_cross32_labels import label_assignment


class Cross32LabelsTests(unittest.TestCase):
    def test_assignment_separates_true_mapping_from_empty_cells(self):
        counts = np.array([[0, 12, 1], [1, 0, 9], [11, 1, 0]])
        m, s = label_assignment(counts)
        self.assertEqual(m.tolist(), [1, 2, 0])
        self.assertEqual(s, 32)


if __name__ == "__main__":
    unittest.main()
