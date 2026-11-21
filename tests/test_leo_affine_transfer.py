import unittest

import numpy as np
import project_paths

from audit_leo_affine_transfer import relation_traces, resource_role


class LeoAffineTransferTests(unittest.TestCase):
    def test_cross_symbol_relation_preserves_missing_constituents(self):
        bits = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
        valid = np.array([[True, True, True], [True, True, True], [True, False, True]])
        trace = relation_traces(bits, valid, (0, 1), (2,), 1)
        self.assertEqual(trace['left'], '11?')
        self.assertEqual(trace['adjusted_right'], '101')
        self.assertEqual(trace['joint'], '10?')

    def test_resource_role_uses_h1_and_word_boundaries(self):
        self.assertEqual(resource_role(227), {'part': 'H1', 'offset': 227})
        self.assertEqual(resource_role(228), {'part': 'Code 5 systematic', 'word': 0, 'offset': 0})
        self.assertEqual(resource_role(1252), {'part': 'Code 5 parity', 'word': 0, 'offset': 1024})
        self.assertEqual(resource_role(2276), {'part': 'Code 5 systematic', 'word': 1, 'offset': 0})
