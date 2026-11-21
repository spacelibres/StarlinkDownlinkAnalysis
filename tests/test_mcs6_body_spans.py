from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_mcs6_body_spans import body_spans, span_from_entry


class MCS6BodySpanTests(unittest.TestCase):
    def test_length_keeps_bits_above_twelve(self):
        self.assertEqual(span_from_entry(bytes.fromhex('a37e301b02')), 8627)
        self.assertEqual(span_from_entry(bytes.fromhex('3300400400')), 68)
        with self.assertRaises(ValueError):
            span_from_entry(bytes.fromhex('33004004'))

    def test_leading_optional_record_and_associated_entry_determine_end(self):
        one = bytes.fromhex('0700 a37e301b02 3300400400')
        two = bytes.fromhex('3300400400 0700 a37e301b02 3300400400')
        self.assertEqual(body_spans(one), [(2, 8627)])
        self.assertEqual(body_spans(two), [(0, 68), (7, 8627)])
        with self.assertRaises(ValueError):
            body_spans(bytes.fromhex('3300400400 3300400400'))
