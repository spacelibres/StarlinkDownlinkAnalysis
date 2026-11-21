import json
import unittest

import numpy as np

import project_paths
from audit_leo_pairs_live import transfer_grid


class LeoPairTransferTests(unittest.TestCase):
    def test_slope_transfers_to_whole_held_frame_groups_and_serializes(self):
        t = np.arange(14)[:, None]/750 + np.arange(600)[None, :]/500000
        slope = -13000.
        phase = np.exp(1j*np.arange(14))[:, None]
        values = phase*np.exp(2j*np.pi*(20*t+.5*slope*t*t))
        rows = transfer_grid(t, values, slope)
        self.assertEqual(len(rows), 6)
        json.dumps(rows)
        for row in rows:
            models = row['models']
            exact, zero = [models[name]['held_explained_fraction'] for name in ('archived', 'zero')]
            self.assertAlmostEqual(exact, 1.)
            self.assertGreater(exact, zero)
