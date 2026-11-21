from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_code5_directory_records import parse_directory


class Code5DirectoryRecordTests(unittest.TestCase):
    def test_little_endian_lengths_and_terminal_after_second_tag(self):
        header = bytes.fromhex('14 00 01 f0 0f 76 08 10 00 ff 60 15 00 08 05 ff ff 00 00 00')
        directory = parse_directory(header)
        self.assertEqual(directory.record_lengths, (135, 21))
        self.assertEqual(directory.padding_bytes, 80)
        self.assertEqual(directory.record_starts, (20, 155))
        self.assertEqual(directory.total_bytes, 256)

    def test_terminal_after_first_tag_keeps_low_twelve_bits(self):
        header = bytes.fromhex('18 00 01 f0 0f 76 08 10 00 ff 60 15 00 01 f0 0f 26 09 80 3a f0 00 00 00')
        directory = parse_directory(header)
        self.assertEqual(directory.record_lengths, (135, 21, 146))
        self.assertEqual(directory.padding_bytes, 58)
        self.assertEqual(directory.total_bytes, 384)

    def test_rejects_truncation_bad_flags_and_extra_directory_bytes(self):
        header = bytearray.fromhex('14 00 01 f0 0f 76 08 10 00 ff 60 15 00 08 05 ff ff 00 00 00')
        corruptions = ((5, 0x77), (13, 0x09), (16, 0))
        for index, value in corruptions:
            with self.subTest(index=index):
                frame = header.copy()
                frame[index] = value
                with self.assertRaises(ValueError):
                    parse_directory(frame)
        with self.assertRaises(ValueError):
            parse_directory(header[:-4])
