from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E094/scripts"))
from audit_code5_orbit_state import accelerations, unpack_state


class Code5OrbitStateTests(unittest.TestCase):
    def test_float_vector_starts_at_high_nibble(self):
        values = (7_000_001.25, -1202.5, 840.75, -12.125, 7480.5, 0.5)
        content = int.from_bytes(struct.pack("<3d3f", *values), "little") << 60
        data = (content | 0x123456789).to_bytes(44, "little")
        position, velocity = unpack_state(data)
        np.testing.assert_array_equal(position, values[:3])
        np.testing.assert_array_equal(velocity, values[3:])

    def test_earth_fixed_correction_has_coriolis_and_centrifugal_signs(self):
        r = np.array([[7e6, 0., 0.]])
        v = np.array([[0., 7200., 0.]])
        a = accelerations(r, v)
        self.assertGreater(a["earth_fixed"][0, 0], a["central"][0, 0])
        self.assertLess(a["reverse_earth_rotation_j2"][0, 0], a["central_j2"][0, 0])
