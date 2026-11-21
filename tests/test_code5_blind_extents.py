import unittest

import numpy as np
import project_paths

from audit_code5_blind_extents import scan_packed, scan_legacy


class Code5BlindExtentTests(unittest.TestCase):
    def test_scan_recovers_start_without_consulting_record_list(self):
        offset = 210
        bit = 8 * offset
        payload = (2 << (bit + 153)) | (256 << (bit + 281)) | (1 << (bit + 297))
        observed = (127 << (bit + 153)) | (4095 << (bit + 281)) | (255 << (bit + 297))
        self.assertEqual(scan_packed(payload, observed, 300), [(210, 40)])
        self.assertEqual(scan_packed(payload, observed & ~(1 << (bit + 297)), 300), [])

    def test_legacy_count_scan_resolves_both_bpsk_signs(self):
        hard = np.full((8, 1024), -1, np.int8)
        offset = 10
        indices = np.r_[112:120, 136:152] + offset * 8
        hard[2, indices] = 1 ^ np.unpackbits(np.array([0x65, 0x60, 0], dtype=np.uint8), bitorder='little')
        self.assertEqual(scan_legacy(hard, np.zeros(1024, dtype=np.uint8)), {(266, 58)})
