import unittest

import project_paths

from audit_code5_slot_bitmap import bitmap_offsets, classify_quartet


class Code5SlotBitmapTests(unittest.TestCase):
    def test_bitmap_can_represent_more_than_one_frame(self):
        self.assertEqual(bitmap_offsets(0), ())
        self.assertEqual(bitmap_offsets(8), (3,))
        self.assertEqual(bitmap_offsets(0b1011), (0, 1, 3))

    def test_whole_quartet_is_classified_against_deadline(self):
        self.assertEqual(classify_quartet(5797, 5830), 'within')
        self.assertEqual(classify_quartet(5829, 5830), 'crossing')
        self.assertEqual(classify_quartet(5832, 5830), 'beyond')
