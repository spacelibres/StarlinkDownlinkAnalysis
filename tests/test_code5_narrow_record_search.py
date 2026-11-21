from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_code5_narrow_record_search import scan_word, select_origin


class Code5NarrowRecordSearchTests(unittest.TestCase):
    def test_search_reports_byte_offset_and_preserves_erasures(self):
        payload = b'\x01' + ((4341 << 11) | 124).to_bytes(4, 'little')
        body = np.full(1024, np.nan)
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='little')
        mask = np.arange(1024, dtype=np.uint8) % 2
        body[48:88] = 1 - 2 * (bits ^ mask[48:88]).astype(float)
        self.assertEqual(scan_word(body, mask, 2), [(262, 4341)])
        body[63] = np.nan
        self.assertEqual(scan_word(body, mask, 2), [])

    def test_origin_requires_agreement_of_multiple_groups_and_unique_winner(self):
        frames = [([(30, 102), (40, 701)], 2, 0), ([(31, 104)], 4, 1), ([(32, 104)], 4, 1)]
        self.assertEqual(select_origin(frames, 1)['origin'], 100)
        self.assertIsNone(select_origin(frames[:1], 1)['origin'])
        frames[1][0].append((40, 703))
        self.assertIsNone(select_origin(frames, 1)['origin'])
