from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import project_paths
sys.path.insert(0,str(ROOT/'experiments/E014/scripts'))
from ut_receiver_core import read_ci16,resample_480,pss_reference,refine_pss,pilot_residual,decode_variable_pdu
from code5_core import PILOTS,TRAIN_PILOTS
from educational_fec_demo import encode_block,crc8,word_bits


class UTCaptureReceiverTests(unittest.TestCase):
    def test_ci16_little_endian_signed_and_truncated_sample_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'test.bin';np.array([[-32768,32767],[123,-456]],dtype='<i2').tofile(p)
            self.assertTrue(np.array_equal(read_ci16(p),[-32768+32767j,123-456j]))
            p.write_bytes(p.read_bytes()+b'0')
            with self.assertRaises(ValueError):read_ci16(p)

    def test_rational_resampling_preserves_complex_tone(self):
        f=53e6;source=np.exp(2j*np.pi*f*np.arange(6000)/250e6).astype(np.complex64)
        result=resample_480(source);expected=np.exp(2j*np.pi*f*np.arange(len(result))/480e6)
        self.assertEqual(len(result),11520)
        self.assertLess(np.max(abs(result[300:-300]-expected[300:-300])),.002)

    def test_pss_local_time_and_frequency_are_fixed_without_crc(self):
        pss=pss_reference();x=np.zeros(8000,complex);start=2000;freq=137000.
        x[start:start+len(pss)]=pss
        x*=np.exp(2j*np.pi*freq*np.arange(len(x))/480e6)
        result=refine_pss(x,start+17,freq)
        self.assertAlmostEqual(result['epoch'],start,places=1)
        self.assertGreater(result['score'],.99)
        self.assertAlmostEqual(result['cfo_hz'],freq,places=2)

    def test_pilot_time_slopes_separate_frequency_and_clock_without_reserved_tones(self):
        slots=np.arange(2,302);time=slots*4.4e-6
        freq=231.;eps=7.5e-6
        phase=2*np.pi*freq*time[:,None]-2*np.pi*234375*PILOTS[None,:]*eps*time[:,None]
        ratios=np.exp(1j*(phase+.2))
        result=pilot_residual(ratios,slots)
        self.assertAlmostEqual(result['frequency_hz'],freq,places=6)
        self.assertAlmostEqual(result['clock_fraction'],eps,places=12)
        changed=ratios.copy();changed[:,np.setdiff1d(np.arange(16),TRAIN_PILOTS)]*=np.exp(2j*time[:,None]*1e5)
        second=pilot_residual(changed,slots)
        self.assertEqual(result,second)

    def test_five_block_six_descriptor_header_uses_prefix_count(self):
        bits=np.zeros(160,dtype=np.uint8);bits[:16]=word_bits(0x34A6)[16:]
        for j in range(6):
            begin=16+j*20;bits[begin:begin+8]=[(5>>i)&1 for i in range(8)]
            bits[begin+8]=1
        checksum=crc8(bits[:-8]);bits[-8:]=[(checksum>>(7-i))&1 for i in range(8)]
        coded=np.concatenate([encode_block(bits[j:j+32]) for j in range(0,160,32)])
        result=decode_variable_pdu(1-2*coded.astype(float),'h2')
        self.assertEqual(result['blocks'],5);self.assertEqual(result['entry_count'],6)
        self.assertTrue(result['crc_ok']);self.assertFalse(result['crc_used_for_selection'])
        self.assertTrue(np.array_equal(result['bits'],bits))


if __name__=='__main__':unittest.main()
