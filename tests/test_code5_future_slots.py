import unittest

import project_paths

from audit_code5_future_slots import cycle_end, decode_nibbles, predict_slots


class Code5FutureSlotsTests(unittest.TestCase):
    def test_nibble_order_and_mask_are_paired(self):
        self.assertEqual(decode_nibbles(bytes.fromhex('1082'), [True] * 8 + [False] + [True] * 7),
                         [0, 1, None, 8])

    def test_reload_horizon_uses_next_boundary_not_extended_wire_count(self):
        codes = [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 2]
        self.assertEqual(cycle_end(5569, 53, 0), 5618)
        predicted = predict_slots(5569, 5618, 5570, 5630, codes, 4, 0)
        self.assertEqual(predicted[0], (1, 5573, 1))
        self.assertEqual(predicted[6], (7, 5597, 0))
        self.assertEqual(predicted[-1], (12, 5617, 1))
        self.assertEqual(len(predicted), 12)

    def test_out_of_capture_and_wrong_time_axis_targets_abstain(self):
        codes = [0, 1, None, 0, 1]
        self.assertEqual(predict_slots(10, 31, 11, 23, codes, 4, -1), [(1, 13, 1), (3, 21, 0)])
