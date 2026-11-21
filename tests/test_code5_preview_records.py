import unittest

import project_paths

from audit_code5_preview_records import lane_positions, transform_preview


class Code5PreviewTests(unittest.TestCase):
    def test_lanes_exclude_partial_fields_at_record_end(self):
        self.assertEqual(lane_positions(35, 161, 60, 12), [161, 221])
        self.assertEqual(lane_positions(37, 161, 60, 12), [161, 221, 281])

    def test_prediction_clears_regular_lanes_and_preserves_terminal_lanes(self):
        size = 48
        word = 3 | (17 << 161) | (4095 << 221) | (256 << 281) | (947 << 341) | (1 << 366)
        expected = word & ~(4095 << 161) & ~(4095 << 221) ^ 2
        self.assertEqual(transform_preview(word.to_bytes(size, 'little'), 161, 60, 12),
                         expected.to_bytes(size, 'little'))
