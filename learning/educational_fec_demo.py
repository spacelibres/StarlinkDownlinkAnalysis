"""Synthetic educational coding exercise, NOT decoding the user's recording.

Known information words are published in the external F250 example:
https://github.com/misaka666/Starlink-Downlink-Analysis
We encode these known words ourselves, introduce errors, decode, and check CRC.
This does not independently validate that paper's over-air receiver convention.
"""
from pathlib import Path
import json
import numpy as np

GENERATORS=(0o155,0o123,0o137)

def word_bits(word):return np.array([int(x) for x in f'{word:032b}'],dtype=np.uint8)

def crc8(bits):
    register=0xff
    for bit in bits:
        feedback=((register>>7)&1)^int(bit)
        register=(register<<1)&0xff
        if feedback:register^=0x4d
    return register

def encode_block(bits):
    state=0;coded=[]
    for bit in np.r_[bits,np.zeros(6,dtype=np.uint8)]:
        register=((state<<1)|int(bit))&127
        coded.extend((register&g).bit_count()%2 for g in GENERATORS)
        state=register&63
    assert state==0
    return np.array(coded,dtype=np.uint8)

def decode_block(coded):
    received=np.asarray(coded).reshape(-1,3);count=len(received)
    metric=np.full(64,np.inf);metric[0]=0
    previous=np.full((count,64),-1,dtype=np.int16);inputs=np.zeros((count,64),dtype=np.uint8)
    for t,observation in enumerate(received):
        following=np.full(64,np.inf)
        for state in range(64):
            if not np.isfinite(metric[state]):continue
            for bit in [0,1]:
                register=((state<<1)|bit)&127;newstate=register&63
                expected=np.array([(register&g).bit_count()%2 for g in GENERATORS])
                cost=metric[state]+np.count_nonzero(expected!=observation)
                if cost<following[newstate]:
                    following[newstate]=cost;previous[t,newstate]=state;inputs[t,newstate]=bit
        metric=following
    state=0;decoded=np.zeros(count,dtype=np.uint8)
    for t in range(count-1,-1,-1):
        decoded[t]=inputs[t,state];state=int(previous[t,state])
        assert state>=0
    assert state==0
    return decoded,float(metric[0])

def run_demo():
    known=np.r_[word_bits(0x3048A020),word_bits(0x000000AD)]
    assert crc8(known[:-8])==0xad
    transmitted=np.r_[encode_block(known[:32]),encode_block(known[32:])]
    corrupted=transmitted.copy();flips=[17,73,155,199];corrupted[flips]^=1
    recovered=[];metrics=[]
    for segment in [corrupted[:114],corrupted[114:]]:
        inputs,metric=decode_block(segment)
        assert np.all(inputs[-6:]==0)
        recovered.extend(inputs[:32]);metrics.append(metric)
    recovered=np.array(recovered,dtype=np.uint8)
    observed_crc=int(''.join(map(str,recovered[-8:])),2)
    computed_crc=crc8(recovered[:-8])
    assert np.array_equal(recovered,known)
    assert observed_crc==computed_crc==0xad
    rebuilt=np.r_[encode_block(recovered[:32]),encode_block(recovered[32:])]
    assert np.array_equal(rebuilt,transmitted)
    return {'data_kind':'synthetic re-encoding of already-published example words; not raw IQ',
            'external_reference_words':['0x3048A020','0x000000AD'],
            'coded_bits':len(transmitted),'deliberately_corrupted_positions':flips,
            'viterbi_hamming_path_costs':metrics,'retained_bits_recovered':len(recovered),
            'computed_crc':f'0x{computed_crc:02X}','known_public_crc':'0xAD',
            'exact_reencoding_matches_synthetic_transmission':True,
            'locally_captured_header_decoded':False}

if __name__=='__main__':
    result=run_demo()
    (Path(__file__).parent/'educational_fec_demo_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
