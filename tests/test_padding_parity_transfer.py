from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E074/scripts"))
from audit_padding_parity_transfer import (
    PREFIX, affine_basis, affine_predictions, nearest_predictions, pack_key, representation,
)


class PaddingParityTransferTests(unittest.TestCase):
    def test_span_prediction_and_outside_span(self):
        cal = [pack_key(np.array([0, 1, 0])),
               pack_key(np.array([1, 0, 0])),
               pack_key(np.array([0, 0, 0]))]
        basis = affine_basis(cal)
        self.assertEqual(representation(pack_key(np.array([1, 1, 0])), basis), 7)
        self.assertEqual(representation(cal[1], basis), 2)
        self.assertIsNone(representation(pack_key(np.array([0, 1, 1])), basis))
        self.assertEqual(len(affine_basis(cal + cal)), 3)

    def test_pack_uses_same_positions_across_a_byte_boundary(self):
        x = np.zeros(9, dtype=np.uint8)
        y = x.copy()
        y[8] = 1
        self.assertNotEqual(pack_key(x), pack_key(y))

    def test_guard_excludes_a_nearby_duplicate_from_both_predictors(self):
        def row(frame, bit):
            return {"frame": frame, "group": frame, "start_resource": 0,
                    "prefix": np.full(PREFIX, bit), "prefix_mask": np.ones(PREFIX, bool),
                    "parity": np.full(4, bit, dtype=np.uint8), "parity_mask": np.ones(4, bool)}

        cal = [row(10, 0), row(40, 1)]
        held = [row(11, 0)]
        permutations = np.array([[0, 1], [1, 0]])
        first = nearest_predictions(cal, held, "prefix", permutations)
        guarded = nearest_predictions(cal, held, "prefix", permutations, guard=3)
        self.assertEqual(first["mean_errors"][0], 0)
        self.assertEqual(guarded["comparisons"][0]["nearest"]["frame"], 40)
        self.assertEqual(affine_predictions(cal, held, "prefix")["checks"][0]["prediction"]["disagreements"], 0)
        self.assertIsNone(affine_predictions(cal, held, "prefix", guard=3)["checks"][0]["prediction"])
