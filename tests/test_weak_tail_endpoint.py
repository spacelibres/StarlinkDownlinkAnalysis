import unittest

import numpy as np

import project_paths
from audit_weak_tail_endpoint import fit_anchor, scan_endpoints
from decode_two_tail_allocations import paths_closing_at


class WeakTailEndpointTests(unittest.TestCase):
    def test_fixed_anchor_and_disjoint_bins_find_an_endpoint_inside_a_slot(self):
        rng = np.random.default_rng(71)
        word = np.where(rng.integers(0, 2, size=60), 1, -1)
        q = np.arange(1004)
        slots = np.arange(20, 42)
        coordinates = slots[:, None]*1004 + q
        end = 22*1004 + 180
        signs = word[(coordinates - end) % 60]
        gain = (1+q/1004)*np.exp(1j*q/550)
        clean = signs * gain * np.exp(.031j*(slots[:, None]-30))
        y = clean*(coordinates >= end) + .15*np.exp(1j*q/400)
        predicted = fit_anchor(y[10:], signs[10:], slots[10:], slots[:6], 1)
        candidates = np.arange(end-180, end+181, 60)
        mask = (q//32) % 2 == 0
        scores = scan_endpoints(y[:6], predicted*signs[:6], coordinates[:6], candidates, mask)
        self.assertEqual(candidates[scores[0].argmax()], end)
        self.assertEqual(candidates[scores[1].argmax()], end)
        y[:6, ~mask] *= -1
        altered = scan_endpoints(y[:6], predicted*signs[:6], coordinates[:6], candidates, mask)
        np.testing.assert_array_equal(scores[0], altered[0])
        self.assertFalse(np.allclose(scores[1], altered[1]))

    def test_two_unobserved_allocations_must_close_the_measured_end(self):
        widths = {5: 2048, 70: 4096, 85: 3277}
        end = 8420 + 228 + 2*4096 + 3277
        layouts = paths_closing_at(end, 8420, widths)
        self.assertIn((2, ((70, 2), (85, 1))), layouts)
        self.assertIn((2, ((85, 1), (5, 4))), layouts)
        self.assertFalse(paths_closing_at(end+1, 8420, widths))
