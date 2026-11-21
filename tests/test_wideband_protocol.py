"""Known-input tests precede real-IQ protocol replay; no archived words used."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E009/scripts'))
sys.path.insert(0,str(ROOT/'learning'))
from educational_fec_demo import encode_block,word_bits
from protocol_core import decode_soft_block,decode_header,header_carriers

class WidebandProtocolTests(unittest.TestCase):
    def test_resource_order_skips_pilots(self):
        carriers=header_carriers()
        self.assertEqual(len(carriers),228)
        self.assertEqual((int(carriers[0]),int(carriers[-1])),(-512,-277))
        self.assertFalse(np.any(np.isin(carriers,np.arange(-496,-488))))

    def test_soft_viterbi_recovers_known_noisy_terminated_word(self):
        rng=np.random.default_rng(8301);bits=rng.integers(0,2,32,dtype=np.uint8)
        coded=encode_block(bits);soft=(2*coded.astype(float)-1)*rng.uniform(.2,2,len(coded))
        soft+=.1*rng.normal(size=len(coded));soft[[15,72]]*=-.2
        decoded=decode_soft_block(soft)
        self.assertTrue(np.array_equal(decoded[:32],bits))
        self.assertTrue(np.all(decoded[32:]==0))

    def test_phase_slope_and_prefix_polarity_do_not_use_crc_to_search(self):
        info=np.r_[word_bits(0x3048A0A0),word_bits(0x000000E8)]
        coded=np.r_[encode_block(info[:32]),encode_block(info[32:])]
        carriers=header_carriers();center=carriers-carriers.mean();rng=np.random.default_rng(8302)
        y=(2*coded.astype(float)-1)*np.exp(1j*(2.4+.109*center))
        y+=.04*(rng.normal(size=len(y))+1j*rng.normal(size=len(y)))
        result=decode_header(y,carriers)
        self.assertEqual(result['words'],[0x3048A0A0,0x000000E8])
        self.assertTrue(result['crc_ok']);self.assertFalse(result['crc_used_for_parameter_selection'])

if __name__=='__main__':unittest.main()
