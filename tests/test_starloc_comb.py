import unittest

import numpy as np

import project_paths
from audit_starloc_narrowband import CombSearch


class StarLocCombTest(unittest.TestCase):
    def test_coherent_grid_transfers_to_new_receiver_without_refitting(self):
        fs = 2_000_000
        samples = fs//20
        t = (np.arange(samples)-(samples-1)/2)/fs
        center, rate, spacing = -32_000., -3_000., 43_945.5
        rng = np.random.default_rng(170)
        tone = np.exp(2j*np.pi*t[:, None]*(center+np.arange(-4, 5)*spacing)).sum(axis=1)
        signal = .20*tone*np.exp(1j*np.pi*rate*t*t)
        search = CombSearch(fs, samples)
        fits = search.fit(signal+rng.standard_normal(samples)+1j*rng.standard_normal(samples))
        rx2 = 1j*signal+rng.standard_normal(samples)+1j*rng.standard_normal(samples)
        moved = search.score(rx2, fits)
        self.assertLess(abs(fits[0]['center_hz']-center), 30)
        self.assertGreater(moved[0]['geometric_ratio'], 6)
        self.assertGreater(moved[0]['geometric_ratio'], max(row['geometric_ratio'] for row in moved[1:]))
