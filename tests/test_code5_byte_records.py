from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E078/scripts"))
from audit_code5_byte_records import AFTER_RECORDS, OPTIONAL_HEADER, erasure_word, optional_structure, structure


class Code5ByteRecordsTests(unittest.TestCase):
    def test_first_byte_sets_both_offsets_and_cross_word_bounds(self):
        b = bytearray(256)
        b[0] = 32
        b[1:11] = bytes.fromhex("00 01 f0 0f 36 0a 10 00 ff 60")
        b[32:39] = bytes.fromhex("00 46 40 00 00 30 70")
        b[98:126] = bytes.fromhex(
            "40 3f 23 11 00 04 00 40 bf 23 11 20 04 00 "
            "40 3f 22 11 40 04 00 40 bf 22 11 60 04 40"
        )
        good = structure(b)
        self.assertEqual(good["record_offset"], 98)
        self.assertTrue(good["full_match"])
        self.assertEqual(good["record_ids"], [0x1123, 0x1123, 0x1122, 0x1122])
        b[0] = 34
        self.assertFalse(structure(b)["full_match"])

    def test_incomplete_record_is_not_promoted(self):
        b = bytearray(128)
        b[0] = 40
        result = structure(b)
        self.assertEqual(result["record_offset"], 106)
        self.assertFalse(result["all_records_available"])
        self.assertEqual(len(result["record_ids"]), 3)
        self.assertFalse(result["full_match"])

    def test_optional_variant_predicts_the_earlier_marker(self):
        data = bytearray(128)
        data[0] = 24
        data[1:11] = OPTIONAL_HEADER
        data[24:31] = bytes.fromhex("00 46 e3 03 00 30 70")
        data[90:100] = AFTER_RECORDS
        parsed = optional_structure(bytes(data), structure(data))
        self.assertEqual(parsed["predicted_next_record"], 90)
        self.assertTrue(parsed["next_record_match"])
        self.assertFalse(parsed["next_record_at_other_position"])

    def test_erased_system_bit_resolved_by_both_folds(self):
        matrix = np.zeros((1024, 1024), np.uint8)
        matrix[1, [7, 145]] = 1
        values = np.ones(2048, dtype=complex)
        values[1] = 1j
        values[1024 + np.array([7, 145])] = -1
        bits, row = erasure_word(values, matrix)
        self.assertEqual(bits[1], 1)
        self.assertEqual(row["systematic_erasures"], 1)
        self.assertTrue(all(f["held_disagreements"] == 0 for f in row["folds"]))
