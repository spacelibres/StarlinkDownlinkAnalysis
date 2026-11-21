import unittest

import numpy as np

import project_paths
from audit_outer_record_lengths import parse_outer


class OuterRecordLengthTests(unittest.TestCase):
    def test_the_second_count_crosses_a_byte_boundary(self):
        header = bytes.fromhex('b8678009c89009')
        parsed = parse_outer(header + bytes(252), 0)
        self.assertEqual(parsed['child_lengths'], [76, 100, 76])
        self.assertEqual(parsed['header_bytes'], 7)
        self.assertEqual(parsed['child_starts'], [7, 83, 183])
        self.assertEqual(parsed['stop'], 259)

    def test_truncation_preserves_header_only_if_requested(self):
        head = bytes.fromhex('904600088100')
        self.assertIsNone(parse_outer(head, 0))
        self.assertIsNone(parse_outer(head[:-1], 0, allow_extent=True))
        self.assertEqual(parse_outer(head, 0, allow_extent=True)['stop'], 134)

    def test_erasure_in_unused_high_nibble_does_not_change_count(self):
        head = bytes.fromhex('904600088100')
        seen = np.ones(48, dtype=bool)
        seen[-4:] = False
        self.assertIsNotNone(parse_outer(head, 0, seen, allow_extent=True))
        seen[32] = False
        self.assertIsNone(parse_outer(head, 0, seen, allow_extent=True))
