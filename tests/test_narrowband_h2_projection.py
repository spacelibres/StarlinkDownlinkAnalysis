import unittest

import numpy as np

import project_paths
from audit_h2_projection import affine_projection, header_bits, visible_positions
from audit_code_observability import encoder_matrix
from educational_fec_demo import crc8


class HeaderProjectionTests(unittest.TestCase):
    def test_geometry_crosses_slot_and_excludes_nondata_grid(self):
        positions = visible_positions(1002, 4, (-512, -2, 510, 511))
        self.assertEqual(positions, (0, 1, 2))

    def test_projection_witness_keeps_affine_crc_and_fixed_bits(self):
        reference = header_bits(["0x34489080", "0x000000CB"])
        free = tuple(range(16, 36))
        result = affine_projection(reference, free, (60, 61, 62))
        self.assertGreater(result["free_dimensions"], 0)
        other = header_bits(result["synthetic_crc_compatible_alternative_hex"])
        self.assertFalse(np.array_equal(other, reference))
        self.assertTrue(np.array_equal(reference[:16], other[:16]))
        self.assertEqual(crc8(other[:-8]), int("".join(map(str, other[-8:])), 2))
        observation = encoder_matrix(2)[[60, 61, 62]]
        self.assertTrue(np.array_equal(observation @ other % 2, observation @ reference % 2))

    def test_full_observation_has_no_ambiguity(self):
        reference = header_bits(["0x34489080", "0x000000CB"])
        result = affine_projection(reference, tuple(range(16, 36)), tuple(range(228)))
        self.assertEqual(result["free_dimensions"], 0)
        self.assertIsNone(result["synthetic_crc_compatible_alternative_hex"])


if __name__ == "__main__":
    unittest.main()
