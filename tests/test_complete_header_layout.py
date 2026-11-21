from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/E010/scripts'))
sys.path.insert(0,str(ROOT/'learning'))
from educational_fec_demo import word_bits,encode_block,crc8
from layout_core import eligible_first_headers,resource_geometry,extract_resources,prefix_fields,second_header_start,decode_second_values

def synthetic_header(prefix,descriptors,blocks):
    bits=np.zeros(32*blocks,dtype=np.uint8);bits[:16]=word_bits(prefix)[16:]
    for index,(mcs,count) in enumerate(descriptors):
        start=16+20*index
        bits[start:start+8]=[(mcs>>i)&1 for i in range(8)]
        bits[start+8:start+20]=[(count>>i)&1 for i in range(12)]
    crc=crc8(bits[:-8]);bits[-8:]=[(crc>>(7-i))&1 for i in range(8)]
    coded=np.concatenate([encode_block(bits[i:i+32]) for i in range(0,len(bits),32)])
    return bits,coded

class CompleteHeaderLayoutTests(unittest.TestCase):
    def test_header_eligibility_recovers_front_partial_frame_but_rejects_tail(self):
        self.assertEqual(eligible_first_headers(11520180),list(range(-1,17)))

    def test_resource_geometry_crosses_slot_without_including_pilots_or_gutter(self):
        slots,q,k=resource_geometry(1000,8)
        self.assertEqual(slots.tolist(),[2]*4+[3]*4)
        self.assertEqual(k.tolist(),[508,509,510,511,-512,-511,-510,-509])
        with self.assertRaises(IndexError):extract_resources(np.zeros(10000,complex),0,0,228)
        with self.assertRaises(ValueError):resource_geometry(-1,8)

    def test_first_header_predicts_location_from_quantity(self):
        for word,expected in [(0x3048A060,12516),(0x3048A0A0,10468),(0x3048A0C0,6372)]:
            bits=np.r_[word_bits(word),word_bits(0)]
            self.assertEqual(second_header_start(bits),expected)
        self.assertEqual(prefix_fields(word_bits(0x34225080))['blocks'],4)

    def test_second_header_uses_prefix_block_count_without_crc_search(self):
        rng=np.random.default_rng(8401)
        cases=[(0x3448,[(9,1)],2),(0x3444,[(21,69),(37,1)],2),
               (0x34CC,[(10,1),(28,50),(68,3)],3),
               (0x3422,[(10,1),(28,50),(37,1),(68,1)],4)]
        for prefix,fields,blocks in cases:
            bits,coded=synthetic_header(prefix,fields,blocks)
            _,_,carriers=resource_geometry(10468,len(coded));phase=1.9+.027*(carriers-carriers[:228].mean())
            values=(2*coded.astype(float)-1)*np.exp(1j*phase)
            values+=.03*(rng.normal(size=len(values))+1j*rng.normal(size=len(values)))
            result=decode_second_values(values,carriers)
            self.assertEqual(result['blocks'],blocks);self.assertTrue(result['crc_ok'])
            self.assertTrue(np.array_equal(result['retained_bits'],bits))
            self.assertFalse(result['crc_used_for_selection'])

    def test_valid_convolutional_code_with_wrong_crc_is_retained_as_failure(self):
        bits,_=synthetic_header(0x34CC,[(10,1),(28,50),(68,3)],3)
        bits[-1]^=1
        coded=np.concatenate([encode_block(bits[i:i+32]) for i in range(0,len(bits),32)])
        _,_,carriers=resource_geometry(10468,len(coded))
        values=(2*coded.astype(float)-1)*np.exp(1j*.4)
        result=decode_second_values(values,carriers)
        self.assertTrue(result['precrc_admitted'])
        self.assertEqual(result['blocks'],3)
        self.assertFalse(result['crc_ok'])
        self.assertEqual(result['coded_hard_disagreements'],0)
        self.assertTrue(np.array_equal(result['retained_bits'],bits))
        self.assertFalse(result['crc_used_for_selection'])

if __name__=='__main__':unittest.main()
