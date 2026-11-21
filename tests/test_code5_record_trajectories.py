from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E092/scripts"))
from audit_code5_record_trajectories import field_information, sliding_values


class Code5RecordTrajectoriesTests(unittest.TestCase):
    def test_read_bit_order_across_byte_boundary(self):
        bank = sliding_values(np.array([[0x00, 0xE0, 0x01]], dtype=np.uint8), 5)
        self.assertEqual(bank[0, 13], 15)

    def test_crc_relation_can_unlock_a_bit_but_is_not_observation(self):
        # Synthetic restriction x0 XOR x1 = 0. Observing x1 reveals x0.
        self.assertEqual(field_information([], [0b11], 0b01), 0)
        self.assertEqual(field_information([0b10], [], 0b01), 0)
        self.assertEqual(field_information([0b10], [0b11], 0b01), 1)
