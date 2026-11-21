"""Conditional two-PDU resource geometry with no CRC-guided branch selection."""
from pathlib import Path
import math
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
import project_paths
from protocol_core import FRAME_START,FRAME_PERIOD,FFT_SIZE,slot_start,fit_blind_phase,decode_soft_block
from receiver_models import pn_reference
from educational_fec_demo import encode_block,crc8

NONPILOT=np.array([k for k in range(-512,512) if k not in [-2,-1,0,1] and not(-496<=k<=-489 or 488<=k<=495)])

def eligible_first_headers(length):
    candidates=range(math.floor(-FRAME_START/FRAME_PERIOD)-1,math.ceil((length-FRAME_START)/FRAME_PERIOD)+1)
    return [f for f in candidates if 0<=slot_start(f,2) and slot_start(f,2)+FFT_SIZE<=length]

def resource_geometry(start,count):
    if start<0 or count<1 or start+count>300*1004:raise ValueError('Resource range outside data slots')
    positions=start+np.arange(count);slots=2+positions//1004;q=positions%1004
    return slots,q,NONPILOT[q]

def extract_resources(iq,frame,start,count):
    slots,q,carriers=resource_geometry(start,count);values=np.empty(count,dtype=complex)
    for slot in np.unique(slots):
        offset=slot_start(frame,int(slot))
        if offset<0 or offset+FFT_SIZE>len(iq):raise IndexError('Requested FFT window is not fully observed')
        spectrum=np.fft.fft(iq[offset:offset+FFT_SIZE].astype(np.complex128));use=slots==slot
        values[use]=spectrum[carriers[use]%FFT_SIZE]
    values*=np.array([pn_reference(int(k),np.array([int(s)]))[0].conjugate() for s,k in zip(slots,carriers)])
    return values,slots,q,carriers

def little(bits):return sum(int(b)<<i for i,b in enumerate(bits))

def prefix_fields(bits):
    bits=np.asarray(bits)
    if len(bits)<16:raise ValueError('Prefix requires 16 retained bits')
    return {'blocks':little(bits[8:11]),'entry_count':little(bits[12:16]),'second_header_role':int(bits[5]),
            'prefix':int(''.join(map(str,bits[:16])),2)}

def parse_descriptors(bits,width=20):
    bits=np.asarray(bits,dtype=np.uint8);prefix=prefix_fields(bits);entries=[]
    stop=16+prefix['entry_count']*width
    if width not in [19,20] or stop>len(bits)-8:return {'width':width,'packing_valid':False,'entries':[]}
    for j in range(prefix['entry_count']):
        data=bits[16+j*width:16+(j+1)*width];index_width=width-12
        entries.append({'mcs_candidate':little(data[:index_width]),'quantity':little(data[index_width:])})
    return {'width':width,'packing_valid':bool(np.all(bits[stop:-8]==0)),
            'zero_residual_bits':int(len(bits)-8-stop),'entries':entries}

def second_header_start(first_bits):
    fields=prefix_fields(first_bits);layout=parse_descriptors(first_bits,20)
    if fields['blocks']!=2 or fields['second_header_role']!=0 or fields['entry_count']!=1 or not layout['packing_valid']:
        raise ValueError('Unsupported first-header working format')
    item=layout['entries'][0]
    if item['mcs_candidate']!=5 or item['quantity']<=0:raise ValueError('First allocation is not frozen code5 model')
    start=228+2048*item['quantity']
    if start+228>300*1004:raise ValueError('Predicted second header outside data grid')
    return start

def decode_second_values(values,carriers):
    values=np.asarray(values,dtype=complex);carriers=np.asarray(carriers,dtype=float)
    if len(values)<228 or len(values)!=len(carriers):raise ValueError('At least two candidate coding blocks required')
    phase,slope=fit_blind_phase(values[:228],carriers[:228]);center=carriers[:228].mean()
    soft=(values*np.exp(-1j*(phase+slope*(carriers-center)))).real
    known=np.r_[np.array([(0x34>>(7-j))&1 for j in range(8)],dtype=np.uint8),np.zeros(24,dtype=np.uint8)]
    reference=2*encode_block(known)[:24].astype(float)-1
    polarity=1 if np.dot(soft[:24],reference)>=0 else -1;soft*=polarity
    first=decode_soft_block(soft[:114]);fields=prefix_fields(first[:32]);blocks=fields['blocks']
    admitted=(fields['prefix']>>8)==0x34 and blocks in [2,3,4] and fields['second_header_role']==1 and 1<=fields['entry_count']<=4
    admitted=bool(admitted and 16+20*fields['entry_count']+8<=32*blocks)
    result={**fields,'precrc_admitted':admitted,'crc_ok':None,'crc_used_for_selection':False,
            'phase_rad':phase,'slope_rad_per_carrier':slope,'polarity':polarity,
            'prefix_coded_correlation':float(np.dot(soft[:24],reference)/max(np.linalg.norm(soft[:24])*np.linalg.norm(reference),1e-30))}
    if not admitted:return result
    needed=114*blocks;result['needed_resources']=needed
    if len(values)<needed:return {**result,'needs_more_resources':True}
    decoded=[decode_soft_block(soft[j*114:(j+1)*114]) for j in range(blocks)]
    bits=np.concatenate([x[:32] for x in decoded]);encoded=np.concatenate([encode_block(bits[j*32:(j+1)*32]) for j in range(blocks)])
    crc=int(''.join(map(str,bits[-8:])),2);computed=crc8(bits[:-8])
    words=[int(''.join(map(str,bits[j:j+32])),2) for j in range(0,len(bits),32)]
    result.update(needs_more_resources=False,crc_ok=crc==computed,computed_crc=computed,observed_crc=crc,
                  retained_bits=bits,soft=soft[:needed],reencoded=encoded,words=words,
                  words_hex=[f'0x{x:08X}' for x in words],
                  coded_hard_disagreements=int(np.sum((soft[:needed]>0).astype(np.uint8)!=encoded)),
                  layout19=parse_descriptors(bits,19),layout20=parse_descriptors(bits,20))
    return result

def decode_second_at(iq,frame,start):
    values,slots,q,k=extract_resources(iq,frame,start,228)
    result=decode_second_values(values,k)
    if result.get('needs_more_resources'):
        values,slots,q,k=extract_resources(iq,frame,start,result['needed_resources'])
        result=decode_second_values(values,k)
    result.update(start_resource=int(start),start_slot=int(slots[0]),start_q=int(q[0]))
    return result
