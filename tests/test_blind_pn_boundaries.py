import unittest

import numpy as np

import project_paths
from audit_blind_pn_boundaries import compare_layout, scan_reference


class BlindPNBoundaryTest(unittest.TestCase):
    def test_partial_symbol_is_excluded_and_word_bounds_are_inferred(self):
        rng = np.random.default_rng(9)
        raw = rng.integers(0, 2, size=(1200, 5), dtype=np.uint8)
        template = rng.integers(0, 2, size=(800, 5), dtype=np.uint8)
        start, cutoff = 191, 411
        raw[start:start + cutoff] = template[:cutoff]
        # Deliberately change the symbol that straddles the bit boundary.
        raw[start + cutoff] = 1 - template[cutoff]
        hits = scan_reference(raw, template, cutoff * 5 + 2, 800)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['first_non_systematic_or_mixed'], start + cutoff)
        self.assertEqual(hits[0]['word_start'], start)
        self.assertEqual(hits[0]['word_stop'], start + 800)
        self.assertEqual(hits[0]['before_match'], 1.0)

    def test_continued_reference_is_rejected(self):
        rng = np.random.default_rng(10)
        raw = rng.integers(0, 2, size=(1200, 4), dtype=np.uint8)
        template = rng.integers(0, 2, size=(800, 4), dtype=np.uint8)
        raw[191:991] = template
        self.assertEqual(scan_reference(raw, template, 411 * 4, 800), [])

    def test_typed_marker_matches_the_header_descriptor_value(self):
        entries = [{'mcs_candidate': 66, 'quantity': 1}, {'mcs_candidate': 70, 'quantity': 1}]
        markers = [{'mcs': 66, 'word_stop': 4106}, {'mcs': 70, 'word_stop': 4106}]
        _, checks = compare_layout(10, entries, markers)
        self.assertEqual(checks[0]['matching_blind_marker_indices'], [0])
        self.assertEqual(checks[0]['role'], 'same_alphabet')
        self.assertEqual(checks[1]['matching_blind_marker_indices'], [])

    def test_near_window_exposes_a_false_early_padding_stop(self):
        rng = np.random.default_rng(48)
        raw = rng.integers(0, 2, size=(1200, 5), dtype=np.uint8)
        template = rng.integers(0, 2, size=(800, 5), dtype=np.uint8)
        raw[191:642] = template[:451]
        hit = scan_reference(raw, template, 411 * 5, 800)[0]
        self.assertLessEqual(hit['after_match'], .68)
        self.assertEqual(hit['near_after_match'], 1.0)
