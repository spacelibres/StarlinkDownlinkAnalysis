import unittest

import numpy as np

import project_paths
from audit_ut_offset_constrained_h2 import coded_bits, constrained_block, decode_structured, evaluate
from educational_fec_demo import crc8, encode_block, word_bits


class UtOffsetConstrainedH2Tests(unittest.TestCase):
    def test_constrained_trellis_matches_brute_force(self):
        rng = np.random.default_rng(18)
        real = rng.normal(size=114)
        mask = np.zeros(32, dtype=np.int8)
        mask[[1, 18, 29]] = -1
        words = np.tile(np.maximum(mask, 0), (8, 1))
        words[:, mask == -1] = ((np.arange(8)[:, None] >> np.arange(3)) & 1)
        score = (1 - 2 * np.array([encode_block(word) for word in words], dtype=float)) @ real
        decoded, metric = constrained_block(real, mask[None])
        np.testing.assert_array_equal(decoded[0], words[score.argmax()])
        self.assertAlmostEqual(metric[0], score.max())

    def test_crc_is_checked_after_decoding_and_never_repairs_bad_crc(self):
        header = np.r_[word_bits(0x3448A8E0), word_bits(0)]
        good = crc8(header[:-8])
        for observed in (good, good ^ 1):
            header[-8:] = word_bits(observed)[-8:]
            soft = 1 - 2 * coded_bits(header).astype(float)
            decoded = decode_structured(soft)
            np.testing.assert_array_equal(decoded, header)
            self.assertEqual(evaluate(decoded, soft)['crc_ok'], observed == good)
