import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/E022/scripts'))
from gap_hypotheses_core import phase_fit, recurrence_fit, recurrence_predict, trajectory


class GapHypothesesTests(unittest.TestCase):
    def test_phase_fit_distinguishes_direction_and_complement(self):
        rng = np.random.default_rng(321)
        seq = rng.integers(0, 2, 31, dtype=np.uint8)
        index = np.arange(125).reshape(5,25)
        bits = trajectory(seq,index,17,-1,True)
        result = phase_fit(seq,index,bits)
        self.assertEqual(result, {'phase':17,'direction':-1,'complement':True,'training_hits':125})

    def test_recurrence_search_fits_small_linear_relation(self):
        rng=np.random.default_rng(17)
        bits=np.zeros((20,60),dtype=np.uint8)
        bits[:,:4]=rng.integers(0,2,(20,4),dtype=np.uint8)
        for i in range(4,60):
            bits[:,i]=bits[:,i-4]^bits[:,i-2]^1
        model=recurrence_fit(bits,order=4)
        self.assertEqual(model['mask'],5)
        self.assertTrue(model['intercept'])
        np.testing.assert_array_equal(recurrence_predict(bits,model,order=4),bits[:,4:])


if __name__=='__main__':
    unittest.main()
