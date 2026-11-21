from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E080/scripts"))
from audit_archive_code5_recovery import checked_single_error, compare_linked_words, normalize_axis


class ArchiveCode5RecoveryTests(unittest.TestCase):
    def test_blind_normalization_preserves_header_polarity_and_ignores_parity(self):
        rng = np.random.default_rng(301)
        signs = rng.choice([-1, 1], size=2048)
        gain = 1.05 * np.exp(.07j)
        values = gain * signs
        values[10] = 5j
        values[1024:] *= 2j
        normalized = normalize_axis(values)
        np.testing.assert_allclose(normalized[:10], signs[:10], atol=1e-12)
        np.testing.assert_allclose(normalized[1024:], 2j * signs[1024:], atol=1e-12)

    def test_missing_systematic_axis_does_not_assume_a_gain(self):
        self.assertIsNone(normalize_axis(np.r_[np.zeros(1024), np.ones(1024)]))

    def test_single_error_requires_both_check_partitions_to_name_the_same_bit(self):
        checks = np.vstack([np.eye(3, dtype=np.uint8)] * 2)
        first = np.array([True, True, True, False, False, False])
        self.assertEqual(checked_single_error(np.array([1, 0, 0]), checks, first), 0)
        self.assertIsNone(checked_single_error(np.array([1, 0, 1]), checks, first))

    def test_shared_observation_excludes_either_words_erasures(self):
        links = {"links": [{"archive_index": 0, "matches": {"0": [{"archive_index": 1}]}}]}
        words = {0: (np.array([7], dtype=np.uint8), np.array([6], dtype=np.uint8)),
                 1: (np.array([0], dtype=np.uint8), np.array([3], dtype=np.uint8))}
        [row] = compare_linked_words(links, words)
        self.assertEqual((row["differing_bits"], row["jointly_observed_bits"],
                          row["jointly_observed_differences"]), (3, 1, 1))
