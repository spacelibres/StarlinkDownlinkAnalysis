from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import project_paths
from audit_code5_directory_checks import crc_bytes
from audit_mcs6_directory_crc import directory_candidates, parse_compact_entries


class MCS6DirectoryCRCTests(unittest.TestCase):
    crc = (0x11F21C5, 0, 0)

    def header(self):
        prefix = bytes.fromhex('14 00 07 03 00 17 02 13 5f e0 05 00 1f 42 ff ff ff')
        return prefix + crc_bytes(prefix, *self.crc)

    def test_crc_recovers_only_declared_erasure(self):
        expected = self.header()
        bits = np.unpackbits(np.frombuffer(expected, dtype=np.uint8), bitorder='little')
        unavailable = np.zeros(len(bits), dtype=bool)
        bits[11] ^= 1
        self.assertEqual(directory_candidates(bits, unavailable, self.crc), [])
        unavailable[11] = True
        self.assertEqual(directory_candidates(bits, unavailable, self.crc), [expected])
        bits[55] ^= 1
        self.assertEqual(directory_candidates(bits, unavailable, self.crc), [])

    def test_compact_tags_stop_at_terminal_field(self):
        labels = parse_compact_entries(self.header())
        self.assertEqual(labels['entry_offsets'], [5, 7])
        self.assertEqual(labels['entry_tags'], [7, 3])
        self.assertEqual(labels['tail_offset'], 12)
        self.assertTrue(labels['tail_and_filler_fit'])
