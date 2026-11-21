import unittest

import numpy as np

import project_paths
from audit_header_parity_lanes import lane_trial, HEADER_RESOURCES, decode_batch, encode_signs
from educational_fec_demo import crc8
from protocol_core import decode_soft_block


class ParityLaneTests(unittest.TestCase):
    def test_hidden_lane_does_not_select_phase_or_bits(self):
        rng = np.random.default_rng(581)
        info = rng.integers(0, 2, 56, dtype=np.uint8)
        bits = np.r_[info, np.unpackbits(np.array([crc8(info)], dtype=np.uint8))]
        waveform = encode_signs(bits)*np.exp(2j)
        self.assertEqual(len(waveform), 2*HEADER_RESOURCES)
        for lane in range(3):
            good = lane_trial(waveform, lane, fixed_blocks=2)
            np.testing.assert_array_equal(good['bits'], bits)
            self.assertTrue(good['crc_ok'])
            self.assertAlmostEqual(good['held_coherence'], 1.)
            corrupt = waveform.copy()
            corrupt[lane::3] *= -1
            bad = lane_trial(corrupt, lane, fixed_blocks=2)
            np.testing.assert_array_equal(bad['bits'], bits)
            self.assertTrue(bad['crc_ok'])
            self.assertAlmostEqual(bad['held_coherence'], -1.)

    def test_batch_matches_existing_terminated_trellis_with_erasures(self):
        rng = np.random.default_rng(582)
        soft = rng.normal(size=(9, HEADER_RESOURCES))
        for i, row in enumerate(soft):
            row[i % 3::3] = 0
        bits, _ = decode_batch(soft)
        expected = np.stack([decode_soft_block(row)[:32] for row in soft])
        np.testing.assert_array_equal(bits, expected)
