from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E008/scripts'))
from cross_receiver_information import overlap_prediction

class CrossReceiverInformationTests(unittest.TestCase):
    def test_unknown_repeated_bits_transfer_across_frame_phase_offsets(self):
        rng=np.random.default_rng(8230);bits=rng.choice([-1,1],(14,11))
        def noisy(gain):return bits*gain[:,None]+.1*(rng.normal(size=bits.shape)+1j*rng.normal(size=bits.shape))
        ga=np.exp(1j*rng.uniform(-np.pi,np.pi,14));gb=np.exp(1j*rng.uniform(-np.pi,np.pi,14))
        result=overlap_prediction(noisy(ga),noisy(ga),noisy(gb),noisy(gb))
        self.assertEqual(result['offset'],0);self.assertGreater(result['test_score'],.9)
        clean=overlap_prediction(bits*ga[:,None],bits*ga[:,None],bits*gb[:,None],bits*gb[:,None])
        self.assertGreater(clean['test_score'],1-1e-10)

    def test_common_reference_does_not_count_as_varying_information(self):
        rng=np.random.default_rng(8231);fixed=rng.choice([-1,1],11)
        def values():return np.exp(1j*rng.uniform(-np.pi,np.pi,14))[:,None]*fixed[None,:]+.1*(rng.normal(size=(14,11))+1j*rng.normal(size=(14,11)))
        result=overlap_prediction(values(),values(),values(),values())
        self.assertLess(abs(result['test_score']),.4)

if __name__=='__main__':unittest.main()
