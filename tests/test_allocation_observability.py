from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import project_paths
sys.path.insert(0,str(ROOT/'experiments/E011/scripts'))
from allocation_core import coherence,detect_suffix,least_squares_change,data_field,predict_end,upward_contrast,unit_identifiability
from layout_core import extract_resources,parse_descriptors
from protocol_core import slot_start
from educational_fec_demo import word_bits


class AllocationObservabilityTests(unittest.TestCase):
    def test_coherence_is_gain_and_global_phase_invariant(self):
        rng=np.random.default_rng(11)
        z=rng.normal(size=320)+1j*rng.normal(size=320)
        self.assertTrue(np.allclose(coherence(z),coherence(3*np.exp(1.7j)*z)))
        self.assertTrue(np.allclose(coherence(np.ones(320,complex)),1))

    def test_data_field_matches_scalar_resource_extractor_and_checks_bounds(self):
        rng=np.random.default_rng(12)
        iq=rng.normal(size=slot_start(-1,301)+2048)+1j*rng.normal(size=slot_start(-1,301)+2048)
        actual,raw,wrong=data_field(iq,-1)
        expected,_,_,_=extract_resources(iq,-1,990,40)
        self.assertTrue(np.allclose(actual[990:1030],expected,rtol=1e-12,atol=1e-12))
        self.assertEqual(actual.shape,(301200,))
        self.assertEqual(raw.shape,wrong.shape)
        with self.assertRaises(IndexError):data_field(iq[:-2],-1)

    def test_suffix_detector_requires_resolvable_terminal_run(self):
        c=np.r_[np.full(160,.12),np.full(80,.8)]
        c[190]=.1
        result=detect_suffix(c)
        self.assertEqual(result['boundary_resource'],160*32)
        self.assertEqual(detect_suffix(np.full(240,.12))['status'],'no_terminal_run')
        self.assertEqual(detect_suffix(np.r_[np.full(224,.12),np.full(16,.8)])['status'],'too_short')
        self.assertEqual(detect_suffix(np.r_[np.full(80,.8),np.full(160,.12)])['status'],'no_terminal_run')
        self.assertFalse(upward_contrast(detect_suffix(np.full(240,.8))))

    def test_single_change_matches_known_truth_and_rejects_constant(self):
        c=np.r_[np.full(200,.12),np.full(80,.8)]
        result=least_squares_change(c)
        self.assertEqual(result['boundary_resource'],200*32)
        self.assertTrue(result['upward_candidate'])
        self.assertFalse(least_squares_change(np.ones(280)*.8)['upward_candidate'])

    def test_resource_unit_prediction_and_unidentifiable_count_rescaling(self):
        fields=[{'mcs_candidate':10,'quantity':1},{'mcs_candidate':28,'quantity':50},
                {'mcs_candidate':37,'quantity':1},{'mcs_candidate':68,'quantity':1}]
        self.assertEqual(predict_end(10468,4,fields)['end_resource'],296995)
        self.assertEqual(predict_end(10468,4,fields,joint_rounding=True)['end_resource'],296962)
        # H1's 19-bit quantity is doubled; its resource unit is not separately identifiable.
        for word in [0x3048A060,0x3048A0A0,0x3048A0C0]:
            bits=np.r_[word_bits(word),word_bits(0)]
            q19=parse_descriptors(bits,19)['entries'][0]['quantity']
            q20=parse_descriptors(bits,20)['entries'][0]['quantity']
            self.assertEqual(q19*1024,q20*2048)
        with self.assertRaises(ValueError):predict_end(10468,4,[{'mcs_candidate':74,'quantity':2}])
        with self.assertRaises(ValueError):predict_end(10468,4,[{'mcs_candidate':10,'quantity':100}])

    def test_negative_minimum_norm_unit_does_not_exclude_positive_solution(self):
        # Counts are nonnegative. Minimum-norm has a negative unit even though
        # units (0.5,0.05,9.95) give the same observations with positive lengths.
        result=unit_identifiability(np.array([[1.,10.,0.],[0.,1.,1.]]),np.array([1.,10.]),minimum_unit=.001)
        self.assertEqual(result['free_unit_dimensions'],1)
        self.assertLess(min(result['unconstrained_minimum_norm_units']),0)
        self.assertLess(result['positive_unit_minimax_error'],1e-7)
        self.assertGreaterEqual(min(result['positive_unit_example']),.001)


if __name__=='__main__':unittest.main()
