from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_code5_later_records import counter_from_record, parse_second_record


class Code5LaterRecordTests(unittest.TestCase):
    def test_second_record_uses_inner_size_and_eight_byte_entries(self):
        item = bytes.fromhex('08 5d 08 00 78 8f 6e 07 28 3f 00 00 00')
        entries = parse_second_record(item)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0], item[4:12])
        for modified in (item[:-1], item[:2] + b'\x09' + item[3:], item[:-1] + b'\x01'):
            with self.assertRaises(ValueError):
                parse_second_record(modified)

    def test_counter_preserves_bit_order_and_ignores_low_flags(self):
        header = b'\x01' + ((11249 << 11) | 0x77F).to_bytes(4, 'little')
        self.assertEqual(counter_from_record(header), 11249)
        self.assertEqual(counter_from_record(b'\x03' + header[1:]), 11249)
        for modified in (header[:-1], b'\x02' + header[1:]):
            with self.assertRaises(ValueError):
                counter_from_record(modified)
