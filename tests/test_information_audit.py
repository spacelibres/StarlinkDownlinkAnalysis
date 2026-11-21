"""Independent positive/null tests for reference removal and pilot prediction."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E008/scripts'))
from receiver_models import pn_reference,cycle_contrast,fit_frame,split_prefix_basis
from audit_code_observability import encoder_matrix,gf2_rank,crc_constraints

class InformationAuditTests(unittest.TestCase):
    def test_erasure_rank_uses_actual_encoder_and_crc_dependencies(self):
        sys.path.insert(0,str(ROOT/'learning'))
        from educational_fec_demo import encode_block,crc8
        bits=np.random.default_rng(8225).integers(0,2,64,dtype=np.uint8)
        matrix=encoder_matrix(2)
        self.assertTrue(np.array_equal((matrix@bits)%2,np.r_[encode_block(bits[:32]),encode_block(bits[32:])]))
        self.assertEqual(gf2_rank(matrix),64)
        crc=crc_constraints(64)
        self.assertEqual(gf2_rank(crc),8)
        self.assertEqual(gf2_rank(matrix[[13,14,15,16,17]]),4)
    def test_prefix_directions_are_noise_orthogonal_with_correct_signal_phase(self):
        rng=np.random.default_rng(8224);p=rng.normal(size=49)+1j*rng.normal(size=49)
        s=rng.normal(size=49)+1j*rng.normal(size=49)
        u,v,rotation=split_prefix_basis(p,s)
        self.assertLess(abs(np.vdot(u,v)),1e-12)
        cross=np.vdot(u,p+s)*np.vdot(v,p+s).conj()*rotation
        self.assertGreater(cross.real,0);self.assertLess(abs(cross.imag),1e-10)
    def test_pn_convention_agrees_with_all_known_edge_pilots(self):
        import project_paths
        import audit_nonpilot_leakage as core
        expected=np.column_stack([pn_reference(k,np.arange(2,302)) for k in range(528,536)])
        ratio=core.expected/expected
        self.assertLess(np.max(abs(ratio-np.exp(-.75j*np.pi))),1e-12)

    def test_constant_pn_and_random_noise_do_not_create_unknown_pattern(self):
        rng=np.random.default_rng(8220)
        noise=.3*(rng.normal(size=(14,18,15))+1j*rng.normal(size=(14,18,15)))
        gain=rng.normal(size=14)+1j*rng.normal(size=14)
        values=gain[:,None,None]+noise
        result=cycle_contrast(values,np.arange(0,18,2),np.arange(1,18,2),rng.uniform(-np.pi,np.pi,(999,9)))
        self.assertLess(abs(result['score']),3)

    def test_unknown_pattern_survives_held_cycle_prediction(self):
        rng=np.random.default_rng(8221)
        bits=rng.choice([-1,1],size=(14,15));gain=np.exp(1j*rng.uniform(-np.pi,np.pi,14))
        values=gain[:,None,None]*bits[:,None,:]+.1*(rng.normal(size=(14,18,15))+1j*rng.normal(size=(14,18,15)))
        result=cycle_contrast(values,np.arange(0,18,2),np.arange(1,18,2),rng.uniform(-np.pi,np.pi,(999,9)))
        self.assertGreater(result['coherence'],.98)
        self.assertGreater(result['score'],3)

    def test_pilot_fit_predicts_reserved_slots_and_recovers_cfo(self):
        rng=np.random.default_rng(8222);fs=2500000.;n=np.arange(50000)
        anchor={'integer_epoch_sample':1000,'fractional_epoch_offset_samples':.2,'fractional_tracking_cfo_hz':-70000.}
        epoch=1000.2;time=(n-epoch)/fs;within=np.mod(time,1/750)
        slot=np.floor(within/4.4e-6).astype(int);local=within-slot*4.4e-6-2/15*1e-6
        pilots=np.exp(.5j*np.pi*rng.integers(0,4,(300,8)))
        carriers=np.arange(527,538);gains=np.exp(1j*rng.uniform(-np.pi,np.pi,len(carriers)))
        unknown=rng.choice([-1,1],size=(300,len(carriers)))
        signal=np.zeros(len(n),complex);active=(slot>=2)&(slot<302)
        for col,k in enumerate(carriers):
            symbols=pilots[slot[active]-2,k-528] if 528<=k<=535 else unknown[slot[active]-2,col]
            signal[active]+=gains[col]*symbols*np.exp(2j*np.pi*(k-531.5)*234375*local[active])/np.sqrt(8)
        signal*=np.exp(2j*np.pi*(-70000+137.5)*n/fs)
        train=((np.arange(300)//15)%2)==0
        fit=fit_frame(signal,anchor,0,pilots,train)
        self.assertLess(abs(fit['residual_cfo_hz']-137.5),.5)
        self.assertLess(fit['heldout_projected_normalized_error'],1e-5)
        interior=fit_frame(signal,anchor,0,pilots,train,edge_guard_samples=2.5)
        self.assertLess(interior['samples_per_symbol'],fit['samples_per_symbol'])
        self.assertLess(interior['heldout_projected_normalized_error'],1e-5)
        self.assertTrue(np.all(np.array(interior['heldout_unit_noise_variance_by_unknown'])>
                               np.array(fit['heldout_unit_noise_variance_by_unknown'])))

    def test_fixed_shared_reference_is_not_frame_varying_information(self):
        rng=np.random.default_rng(8223);gain=np.exp(1j*rng.uniform(-np.pi,np.pi,14))
        fixed=rng.choice([-1,1],15)
        values=gain[:,None,None]*fixed[None,None,:]+.1*(rng.normal(size=(14,18,15))+1j*rng.normal(size=(14,18,15)))
        phases=rng.uniform(-np.pi,np.pi,(999,9))
        result=cycle_contrast(values,np.arange(0,18,2),np.arange(1,18,2),phases,remove_shared=True)
        self.assertLess(abs(result['score']),3)

if __name__=='__main__':unittest.main()
