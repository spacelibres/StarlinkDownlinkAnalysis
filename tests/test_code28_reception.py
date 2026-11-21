import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import project_paths
sys.path.insert(0, str(ROOT / 'experiments/E021/scripts'))
from code28_receiver_core import crossfit_gain, residual_sample_scale
from audit_code28_reencoding import fft_binary_parities
from code21_ring import binary_blocks, evaluate, pack_blocks


class RefinedCode28ReceiverTests(unittest.TestCase):
    def test_timing_correction_undoes_late_sampling(self):
        eps = 31e-6
        radians_per_carrier_per_slot = 2*np.pi*2112*eps/2048
        self.assertAlmostEqual(residual_sample_scale(radians_per_carrier_per_slot), -eps)

    def test_gain_of_target_is_fitted_on_other_selector_parity(self):
        observed = np.array([1.2+.1j, 1.2+.1j, -1.2-.1j, -1.2-.1j])
        gain_group = np.zeros(4, dtype=int)
        selector = np.array([0, 1, 0, 1])
        points = np.array([-1., 1.])
        normalized = crossfit_gain(observed, gain_group, selector, points)
        np.testing.assert_allclose(normalized, [1, 1, -1, -1])
        changed = observed.copy()
        changed[0] *= 1.5
        alternate = crossfit_gain(changed, gain_group, selector, points)
        np.testing.assert_allclose(alternate[0], 1.5)

    def test_independent_cyclic_convolution_matches_exact_polynomial_math(self):
        generator = [[3, 9], [5, 2]]
        u = np.array([[1, 1, 0, 1, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 1, 0]], dtype=np.uint8)
        seed = binary_blocks(generator, 4).reshape(2, 2, 4)
        result, residual = fft_binary_parities(u.reshape(-1, 2, 4), seed)
        np.testing.assert_array_equal(result, binary_blocks(evaluate(pack_blocks(u, 4), generator, 4), 4))
        self.assertLess(residual, 1e-10)


if __name__ == '__main__':
    unittest.main()
