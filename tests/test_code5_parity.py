from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import project_paths
sys.path.insert(0,str(ROOT/'experiments/E012/scripts'))
from code5_core import sinc_sample,recover_generator,qc_augment,equalize_grid,reference_gain,LOADED,PILOTS,TRAIN_PILOTS
from receiver_models import pn_reference


class Code5ParityTests(unittest.TestCase):
    def test_signal_gain_is_fixed_by_known_reference_not_target_payload(self):
        reference=np.array([1.,-1.,1.,1.,-1.,-1.])
        gain=.6*np.exp(.07j)
        self.assertAlmostEqual(abs(reference_gain(reference*gain,reference)-gain),0,places=12)
        # Recovered gain predicts a separately supplied sign sequence.
        target=np.array([-1.,1.,-1.])
        prediction=reference_gain(reference*gain,reference)*target
        self.assertTrue(np.allclose(prediction,gain*target))

    def test_fractional_complex_tone_sampling_and_bounds(self):
        values=np.exp(2j*np.pi*.23*np.arange(4096))
        points=np.arange(50,4050)+.371
        self.assertLess(np.max(abs(sinc_sample(values,points)-np.exp(2j*np.pi*.23*points))),.002)
        with self.assertRaises(IndexError):sinc_sample(values,np.array([.1]))

    def test_binary_recovery_predicts_new_inputs_and_rejects_missing_or_inconsistent_rank(self):
        rng=np.random.default_rng(1201);p=rng.integers(0,2,(16,12),dtype=np.uint8)
        u=np.r_[np.eye(16,dtype=np.uint8),rng.integers(0,2,(10,16),dtype=np.uint8)]
        v=(u@p)%2;fit=recover_generator(u,v)
        self.assertEqual(fit['rank'],16);self.assertEqual(fit['inconsistent_equations'],0)
        self.assertTrue(np.array_equal(fit['generator'],p))
        fresh=rng.integers(0,2,(20,16),dtype=np.uint8)
        self.assertTrue(np.array_equal((fresh@fit['generator'])%2,(fresh@p)%2))
        self.assertIsNone(recover_generator(u[:4],v[:4])['generator'])
        altered=v[:1].copy();altered[0,0]^=1
        bad=recover_generator(np.r_[u,u[:1]],np.r_[v,altered])
        self.assertEqual(bad['inconsistent_equations'],1);self.assertIsNone(bad['generator'])

    def test_qc_augmentation_preserves_known_circulant_forward_model(self):
        rng=np.random.default_rng(1202);block=np.array([1,1,0,1],dtype=np.uint8)
        p=np.stack([np.roll(block,j) for j in range(4)])
        u=rng.integers(0,2,(3,4),dtype=np.uint8);v=(u@p)%2
        a,b=qc_augment(u,v,4)
        self.assertEqual(a.shape,(12,4));self.assertTrue(np.array_equal((a@p)%2,b))

    def test_reserved_pilots_cannot_change_calibration(self):
        rng=np.random.default_rng(1203);slots=np.arange(2,6)
        k=LOADED;pn=np.column_stack([pn_reference(int(c),slots) for c in k])
        h=(1+.0001*k)*np.exp(1j*.002*k)
        symbols=(1-2*rng.integers(0,2,pn.shape))*pn
        pidx=np.array([np.flatnonzero(k==c)[0] for c in PILOTS]);symbols[:,pidx]=pn[:,pidx]
        ramps=np.exp(1j*(.4+slots[:,None]*.02+.0004*k[None,:]))
        sss=np.ones(len(k),complex);received=symbols*h[None,:]*ramps
        first=equalize_grid(received,h,sss,slots)
        reserved=np.setdiff1d(np.arange(16),TRAIN_PILOTS)
        changed=received.copy();changed[:,pidx[reserved]]*=-1
        second=equalize_grid(changed,h,sss,slots)
        self.assertTrue(np.allclose(first['calibration'],second['calibration']))
        self.assertLess(first['reserved_pilot_evm'],.02)
        self.assertGreater(second['reserved_pilot_evm'],1.9)
        expected=symbols[:,np.isin(k,PILOTS,invert=True)]*pn[:,np.isin(k,PILOTS,invert=True)].conj()
        self.assertLess(np.max(abs(first['nonpilot'][:,16:-16]-expected[:,16:-16])),.02)


if __name__=='__main__':unittest.main()
