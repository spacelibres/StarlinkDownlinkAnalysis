import unittest

import numpy as np

import project_paths
from audit_narrowband_cycle import ambiguity, apply_covariance, bank, covariance, projectors_for_reference


class CycleProjectionTests(unittest.TestCase):
    def test_ambiguity_preserves_composition_by_lane(self):
        template = np.zeros(60, dtype=bool)
        template[[0, 4, 1]] = True
        result = ambiguity(template, np.array([0, 8, 13]))
        self.assertEqual(result["same_lane_composition_compatible_templates"], 13 * 14)

    def test_projector_rejects_unknown_constants(self):
        indices = np.arange(15)[:, None] * 44
        template = np.zeros(60)
        template[[0, 8, 20]] = 1
        vectors = bank(template, indices)
        result = apply_covariance(vectors, covariance(np.full((3, 15, 1), 2 + 1j)))
        self.assertEqual(float(result.max()), 0.0)

    def test_projector_retains_pattern_under_complex_gain(self):
        indices = np.arange(15)[:, None] * 44
        template = np.zeros(60)
        template[[0, 8, 20]] = 1
        vectors = bank(template, indices)
        data = template[(indices + 4) % 60] * (2 - 3j) - 5j
        result = apply_covariance(vectors, covariance(data[None, :, :]))
        self.assertAlmostEqual(float(result[4]), 1.0)

    def test_published_reference_ratio_is_projected_out(self):
        ratio = np.array([[1], [-1], [1], [-1], [1]], dtype=complex)
        projector = projectors_for_reference(ratio)[0]
        np.testing.assert_allclose(projector @ (5 + (2 - 3j) * ratio[:, 0]), 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(projector @ np.array([1, 0, -1, 0, 0])), 1)


if __name__ == "__main__":
    unittest.main()
