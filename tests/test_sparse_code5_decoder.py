from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import project_paths
sys.path.insert(0,str(ROOT/'experiments/E013/scripts'))
from sparse_code5_core import find_sparse_checks,independent_checks,binary_rank,normalized_min_sum,syndrome
from code5_core import recover_generator


class SparseCode5Tests(unittest.TestCase):
    def test_sparse_search_recovers_full_dual_of_known_circulant_code(self):
        eye=np.eye(4,dtype=np.uint8)
        a=eye^np.roll(eye,1,axis=1)^np.roll(eye,3,axis=1)
        b=eye^np.roll(eye,1,axis=1)^np.roll(eye,2,axis=1)
        zero=np.zeros_like(a)
        hu=np.block([[a,zero],[zero,a]]);hp=np.block([[b,zero],[zero,b]])
        p=recover_generator(hp,hu)['generator'].T
        found=find_sparse_checks(p,lift=4,max_weight=6)
        selected=independent_checks(found['seeds'],8,16,4)
        h=selected['matrix'];g=np.c_[np.eye(8,dtype=np.uint8),p]
        self.assertEqual(binary_rank(h),8)
        self.assertTrue(np.all((g@h.T)%2==0))
        self.assertLessEqual(int(h.sum(axis=1).max()),6)
        self.assertTrue(np.array_equal(recover_generator(h[:,8:],h[:,:8])['generator'].T,p))

    def test_weak_error_is_corrected_and_positive_llr_scaling_is_invariant(self):
        h=np.array([[1,1,1,0,1,0,0],[1,1,0,1,0,1,0],[1,0,1,1,0,0,1]],dtype=np.uint8)
        llr=np.full(7,8.);llr[0]=-.4
        result=normalized_min_sum(llr,h)
        self.assertTrue(result['converged']);self.assertTrue(np.all(result['bits']==0))
        self.assertGreater(result['iterations'],0)
        scaled=normalized_min_sum(llr*3,h)
        self.assertTrue(np.array_equal(result['bits'],scaled['bits']))
        self.assertEqual(result['iterations'],scaled['iterations'])

    def test_zero_syndrome_does_not_prove_original_word_was_recovered(self):
        h=np.array([[1,1,1,0,1,0,0],[1,1,0,1,0,1,0],[1,0,1,1,0,0,1]],dtype=np.uint8)
        result=normalized_min_sum(np.full(7,-8.),h)
        self.assertTrue(result['converged']);self.assertEqual(result['iterations'],0)
        self.assertTrue(np.all(result['bits']==1))
        self.assertTrue(np.all(syndrome(result['bits'][None,:],h)==0))
        self.assertNotEqual(int(np.sum(result['bits'])),0)


if __name__=='__main__':unittest.main()
