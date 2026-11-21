import unittest

import numpy as np

import project_paths
from audit_mcs85_records import detect_records, predict_successor


class MCS85RecordSpacingTests(unittest.TestCase):
    def test_detection_can_cross_a_word_and_does_not_read_verification_bytes(self):
        data = np.zeros(64, dtype=np.uint8)
        value = np.arange(1, 21, dtype=np.uint8)
        detect = np.where(np.arange(20) % 2 == 0, 255, 0).astype(np.uint8)
        verify = 255 - detect
        data[23:43] = value
        data[24] ^= 4
        rows = detect_records(data, value, detect, verify, max_distance=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['byte_offset'], 23)
        self.assertEqual(rows[0]['verification_bit_disagreements'], 1)

    def test_prediction_requires_the_exact_coordinate_and_retains_censoring(self):
        offsets = {28, 38}
        self.assertEqual(predict_successor(14, 14, 60, offsets), 'matched')
        self.assertEqual(predict_successor(14, 13, 60, offsets), 'not_detected')
        self.assertEqual(predict_successor(38, 4, 60, offsets), 'outside')
