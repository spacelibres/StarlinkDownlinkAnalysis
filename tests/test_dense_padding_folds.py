from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E077/scripts"))
from audit_dense_padding_folds import fit_dense, fold_word, walsh_transform


class DensePaddingFoldTests(unittest.TestCase):
    def test_residue_fold_and_erasure(self):
        words = np.array([[1, 0, 1, 1, 0, 1, 1, 0]], dtype=np.uint8)
        visible = np.ones(words.shape, bool)
        visible[0, 1] = False
        bits, mask = fold_word(words, visible, 4, 2)
        np.testing.assert_array_equal(bits, [[0, 1, 1, 1]])
        np.testing.assert_array_equal(mask, [[True, False, True, True]])

    def test_dense_relation_and_unidentified_coefficient(self):
        points = np.arange(16, dtype=int)
        desired = np.array([int(x & 15).bit_count() % 2 ^ 1 for x in points], dtype=np.uint8)[:, None]
        full = np.ones(desired.shape, bool)
        fit = fit_dense(points, desired, full, np.ones(len(points)), 4)
        self.assertEqual(fit[0]["coefficients"].tolist(), [15])
        self.assertEqual(fit[0]["positive"].tolist(), [False])
        ambiguous = fit_dense(points[::2], desired[::2], full[::2], np.ones(8), 4)
        self.assertEqual(ambiguous[0]["coefficients"].tolist(), [14, 15])

    def test_walsh_matches_expected_sign_convention(self):
        np.testing.assert_array_equal(walsh_transform(np.arange(4.)), [6., -2., -4., 0.])
