from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E096/scripts"))
from audit_code5_fleet_identity import IRREDUCIBLE_BITS, ID_OFFSET, fleet_number, name_number


class Code5FleetIdentityTests(unittest.TestCase):
    def test_fleet_field_crosses_a_byte_boundary_and_ignores_adjacent_bits(self):
        width = ID_OFFSET + IRREDUCIBLE_BITS
        word = (1 << width) | (31848 << ID_OFFSET) | ((1 << ID_OFFSET) - 1)
        self.assertEqual(fleet_number(word), 31848)

    def test_catalog_name_uses_fleet_suffix(self):
        self.assertEqual(name_number("STARLINK-2472"), 2472)
        with self.assertRaises(ValueError):
            name_number("STARLINK-2472 DEB")
