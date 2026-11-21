import unittest

import project_paths

from audit_code5_record_extents import read_extent, read_masked_extent


class Code5RecordExtentTests(unittest.TestCase):
    def test_count_and_footer_decode_without_outer_boundary(self):
        head = bytearray(50)
        head[0] = 3
        head[17:19] = bytes.fromhex('10e0')
        head[19] = 4
        header = int.from_bytes(head, 'little') | (256 << 281) | (1 << 297)
        stream = header.to_bytes(50, 'little')
        result = read_extent(stream)
        self.assertEqual((result['count'], result['marker_bit'], result['footer_octets'], result['bytes']),
                         (2, 281, 1, 40))
        self.assertIsNone(read_extent(bytes(head)))

    def test_empty_and_legacy_layouts_use_different_count_conventions(self):
        empty = bytes([1]) + bytes(34)
        self.assertEqual(read_extent(empty)['bytes'], 20)
        old = bytearray(64)
        old[0], old[14], old[17] = 1, 0x65, 0x60
        self.assertEqual((read_extent(old)['count'], read_extent(old)['bytes']), (6, 58))

    def test_masked_read_uses_only_count_marker_and_footer_length(self):
        stream = (4 << 152 | 256 << 281 | 1 << 297).to_bytes(50, 'little')
        available = set(range(153, 160)) | set(range(281, 293)) | set(range(297, 305))
        self.assertEqual(read_masked_extent(stream, available.__contains__), 40)
        available.remove(297)
        self.assertIsNone(read_masked_extent(stream, available.__contains__))
