import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import project_paths
sys.path.insert(0, str(ROOT / 'experiments/E020/scripts'))
from quotient_lift_core import pascal_transform, fit_layers
from soft_quotient_core import expected_check_agreement, label_parity_means
from code21_ring import binary_blocks, evaluate, pack_blocks, reduce_polynomial


class QuotientLiftTests(unittest.TestCase):
    def test_pascal_is_self_inverse_and_has_evaluation_at_one(self):
        source = np.array([[0, 0, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        transformed = pascal_transform(source)
        np.testing.assert_array_equal(transformed, [[1, 1, 1, 1], [0, 0, 1, 0]])
        np.testing.assert_array_equal(pascal_transform(transformed), source)

    def test_layered_fit_recovers_known_small_qc_generator(self):
        rng = np.random.default_rng(12)
        u = rng.integers(0, 2, (64, 8), dtype=np.uint8)
        p = [[5, 3], [8, 2]]
        v = binary_blocks(evaluate(pack_blocks(u, 4), p, 4), 4)
        model, info, predictions = fit_layers(pascal_transform(u.reshape(-1, 2, 4)),
                                              pascal_transform(v.reshape(-1, 2, 4)),
                                              np.arange(len(u)) < 40)
        np.testing.assert_array_equal(model, pascal_transform(binary_blocks(p, 4).reshape(2, 2, 4)))
        np.testing.assert_array_equal(predictions, pascal_transform(v.reshape(-1, 2, 4)))
        self.assertEqual(info[0]['tied_best_counts'], [1, 1])

    def test_truncated_transform_agrees_with_projection_then_transform(self):
        data = np.array([[1, 0, 1, 0, 0, 1, 1, 1]], dtype=np.uint8)
        folded = [[reduce_polynomial(pack_blocks(data, 8)[0][0], 4)]]
        np.testing.assert_array_equal(pascal_transform(data)[..., :4],
                                      pascal_transform(binary_blocks(folded, 4)))

    def test_soft_check_score_handles_symbols_shared_by_two_input_blocks(self):
        probabilities = np.array([[[.05, .1, .05, .1, .1, .2, .2, .2],
                                   [.2, .1, .2, .1, .1, .1, .1, .1]]])
        means = label_parity_means(probabilities)
        base = np.array([2, 3])
        toggles = np.array([[4, 0], [2, 1]])
        options = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        calculated = expected_check_agreement(means, base, toggles, options)
        expected = []
        for a, b in options:
            m = [2 ^ (4*a) ^ (2*b), 3 ^ b]
            expected.append(sum(probabilities[0, 0, x] * probabilities[0, 1, y]
                                * (-1)**(int(x & m[0]).bit_count()+int(y & m[1]).bit_count())
                                for x in range(8) for y in range(8)))
        np.testing.assert_allclose(calculated[0], expected, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
