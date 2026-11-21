import unittest

import numpy as np
import project_paths
from audit_cross32_spare_lane import joint_lane, paired_zero_counts


class Cross32SpareLaneTests(unittest.TestCase):
    def test_joint_and_separate_boundaries_first_diverge_at_fifth_word(self):
        self.assertEqual([joint_lane(i) for i in range(11)],
                         [16384, 16384, 16384, 16384, 16379, 16379, 16379, 16379, 16379, 16374, 16374])

    def test_paired_zero_counts_exclude_either_erasure_and_retain_disagreement(self):
        bits = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.uint8)
        mask = np.array([[1, 1], [1, 1], [1, 0], [0, 1]], dtype=bool)
        self.assertEqual(paired_zero_counts(bits[:, 0], bits[:, 1], mask.all(axis=1)), [[0, 1], [1, 0]])
