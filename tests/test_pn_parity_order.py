from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E073/scripts"))
from audit_pn_parity_order import byte_run, parity_intervention


class PnParityOrderTests(unittest.TestCase):
    def test_byte_run_has_complete_exact_bytes_and_respects_input_extent(self):
        mask = np.zeros(56, dtype=np.uint8)
        mask[[0, 19, 45]] = 1
        self.assertEqual(byte_run(mask), (24, 40))
        self.assertEqual(byte_run(mask[0:31]), (8, 16))

    def test_parity_intervention_retains_received_parity_and_detects_kernel(self):
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        system = np.array([1, 1, 1], dtype=np.uint8)
        received = np.array([1, 1, 1], dtype=np.uint8)
        result = parity_intervention(system, received, matrix, slice(1, 3), {
            "zero": np.array([0, 0], dtype=np.uint8),
            "kernel": np.array([1, 0], dtype=np.uint8),
        })
        self.assertEqual(result["observed"], {"mismatches": 1, "changed_predictions": 0})
        self.assertEqual(result["zero"], {"mismatches": 2, "changed_predictions": 1})
        self.assertEqual(result["kernel"], result["observed"])
