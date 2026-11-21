import unittest

import numpy as np

import project_paths
from audit_wav_tail_support import endpoint_support, first_supported_slots, select_by_support, tail_columns
from audit_wav_tessellation import CARRIERS, forward


class WavTailSupportTests(unittest.TestCase):
    def test_exclusive_endpoint_includes_equality_and_carries_across_slots(self):
        q = np.array([502, 503])
        ends = np.array([0, 502, 503, 504, 1506])
        np.testing.assert_array_equal(first_supported_slots(ends, q),
                                      [[2, 2], [2, 2], [3, 2], [3, 3], [3, 3]])

    def test_equivalent_endpoints_retain_all_members_and_phase(self):
        q = np.array([502, 503])
        starts, groups = endpoint_support(8, q)
        flattened = np.sort(np.concatenate(groups))
        np.testing.assert_array_equal(flattened, np.arange(8, 301200, 60))
        for row, values in zip(starts, groups, strict=True):
            np.testing.assert_array_equal(first_supported_slots(values, q), np.broadcast_to(row, (len(values), 2)))

    def test_bank_matches_frozen_forward_model_at_slot_nine(self):
        harmonics = np.r_[np.arange(-500, -460), np.arange(200, 270)]
        symbols = np.where(np.arange(60).reshape(4, 15) % 3, 1, -1)
        geom = np.array([[9, 9, 9, 9], [302, 302, 302, 302]])
        offset = 117_180.
        bank = tail_columns(harmonics, offset, CARRIERS, symbols, geom)
        prior = forward(harmonics, offset).reshape(len(harmonics), 4, 15)
        np.testing.assert_allclose(bank[0], np.einsum('nks,ks->nk', prior, symbols), atol=2e-12)
        np.testing.assert_array_equal(bank[1], np.zeros_like(bank[1]))

    def test_earliest_interval_is_compared_with_its_own_best_late_candidate(self):
        chosen = select_by_support(np.array([.2, .1, .3, .6]), np.array([True, True, False, False]))
        self.assertEqual(chosen, {'selected': 3, 'best_early': 0, 'best_late': 3})
