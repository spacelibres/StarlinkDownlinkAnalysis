from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E091/scripts"))
from audit_code5_directory_checks import crc_bytes, fit_affine


class Code5DirectoryCheckTests(unittest.TestCase):
    def test_reflected_trailer_and_held_length_prediction(self):
        prefix = bytes.fromhex("18 00 01 f0 0f 36 0a 10 00 ff 60 15 00 01 f0 0f 26 0f 80 3e f0")
        poly, seed, residual = 0x11F21C5, 0xF4CAD4, 0x530AC1
        self.assertEqual(crc_bytes(prefix, poly, seed, residual), bytes.fromhex("3d69c6"))
        words = []
        for length in (24, 32):
            data = bytes([length]) + bytes(range(length - 4))
            words.append(data + crc_bytes(data, poly, seed, residual))
        affine = fit_affine(words, poly)
        self.assertEqual(affine["rank"], 22)
        self.assertEqual(crc_bytes(prefix, poly, affine["seed"], affine["residue"]), bytes.fromhex("3d69c6"))
        self.assertIsNone(fit_affine(words, poly, start=1))
