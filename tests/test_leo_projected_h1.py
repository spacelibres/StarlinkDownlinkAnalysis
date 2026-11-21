import unittest

import numpy as np
import project_paths

from audit_leo_projected_h1 import minimum_distance_prefix


class LeoProjectedH1Tests(unittest.TestCase):
    def test_projection_keeps_ties_and_ignores_erased_coordinates(self):
        candidates = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 0, 0]], dtype=np.uint8)
        tied = minimum_distance_prefix(candidates, np.array([0, -1, -1]), 1)
        self.assertEqual(tied, {'distance': 0, 'messages': [0, 1], 'signs': [0], 'bits': '?'})
        unique = minimum_distance_prefix(candidates, np.array([1, 0, 0]), 1)
        self.assertEqual(unique, {'distance': 0, 'messages': [1], 'signs': [1], 'bits': '1'})
