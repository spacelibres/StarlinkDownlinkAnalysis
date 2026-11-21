"""Frozen 480 MS/s legacy receiver geometry and soft header decoding.

The blind BPSK objective follows the user's archived kulib/55 method. Timing,
PN, generators and CRC were established on the historical capture; replay is
not a new independent protocol discovery. CRC is a check, never a selector.
"""
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'learning'))
import project_paths
from receiver_models import pn_reference
from educational_fec_demo import encode_block,word_bits,crc8

FRAME_START=639992
FRAME_PERIOD=640010
SYMBOL=2112
FFT_SIZE=2048
CP=64
EPS=(FRAME_PERIOD-640000)/640000
GENERATORS=(0o155,0o123,0o137)
KNOWN_PREFIX=0x3048

def header_carriers():
    return np.array([k for k in range(-512,-276) if not -496<=k<=-489],dtype=int)

def slot_start(frame,slot):
    return int(round(FRAME_START+frame*FRAME_PERIOD+slot*SYMBOL*(1+EPS)))+CP

def frame_count(length):
    count=0
    while slot_start(count,301)+FFT_SIZE<=length:count+=1
    return count

def decode_soft_block(soft):
    soft=np.asarray(soft,dtype=float)
    if soft.shape!=(114,) or not np.isfinite(soft).all():raise ValueError('Expected 114 finite soft values')
    states=np.arange(64);inputs=np.arange(2)
    register=(states[:,None]<<1)|inputs[None,:]
    signs=np.stack([np.vectorize(lambda x:(int(x)&g).bit_count()%2)(register) for g in GENERATORS],axis=-1)*2.-1.
    metric=np.full(64,-np.inf);metric[0]=0
    history=np.zeros((38,64),dtype=bool)
    for time,observation in enumerate(soft.reshape(38,3)):
        candidates=metric[:,None]+np.einsum('suj,j->su',signs,observation)
        left=candidates[states>>1,states&1];right=candidates[(states>>1)+32,states&1]
        history[time]=right>left;metric=np.maximum(left,right)
    state=0;decoded=np.zeros(38,dtype=np.uint8)
    for time in range(37,-1,-1):
        decoded[time]=state&1;state=(state>>1)+(32 if history[time,state] else 0)
    if state!=0:raise RuntimeError('Invalid initial trellis state')
    return decoded

def fit_blind_phase(y,carriers):
    """Frozen legacy |Re| objective; no CRC or decoded-bit feedback."""
    centered=carriers-carriers.mean();slopes=np.arange(-.16,.1600001,.002)
    z=y[None,:]*np.exp(-1j*slopes[:,None]*centered)
    origins=.5*np.angle(np.sum(z*z,axis=1));offset=np.linspace(-1.4,1.4,15)
    value=np.sum(abs((z[:,None,:]*np.exp(-1j*(origins[:,None]+offset)[...,None])).real),axis=-1)
    i,j=np.unravel_index(np.argmax(value),value.shape);phase=origins[i]+offset[j];slope=slopes[i]
    for _ in range(2):
        phases=np.linspace(phase-.05,phase+.05,21);slopes=np.linspace(slope-.002,slope+.002,21)
        rotated=y[None,None,:]*np.exp(-1j*(phases[:,None,None]+slopes[None,:,None]*centered))
        value=np.sum(abs(rotated.real),axis=-1);i,j=np.unravel_index(np.argmax(value),value.shape)
        phase=phases[i];slope=slopes[j]
    return float(phase),float(slope)

def decode_header(y,carriers):
    y=np.asarray(y,dtype=complex);carriers=np.asarray(carriers,dtype=float)
    phase,slope=fit_blind_phase(y,carriers)
    soft=(y*np.exp(-1j*(phase+slope*(carriers-carriers.mean())))).real
    # First 48 encoded resources depend only on the pre-existing 16-bit prefix.
    # Freeze this authority explicitly; do not choose polarity by trying CRCs.
    prefix=np.array([(KNOWN_PREFIX>>(15-j))&1 for j in range(16)],dtype=np.uint8)
    reference=2*encode_block(np.r_[prefix,np.zeros(16,dtype=np.uint8)])[:48].astype(float)-1
    polarity=1 if np.dot(soft[:48],reference)>=0 else -1
    soft*=polarity
    blocks=[decode_soft_block(soft[:114]),decode_soft_block(soft[114:])]
    bits=np.r_[blocks[0][:32],blocks[1][:32]]
    words=[int(''.join(map(str,row)),2) for row in bits.reshape(2,32)]
    computed=crc8(bits[:-8]);observed=words[1]&255
    encoded=np.r_[encode_block(bits[:32]),encode_block(bits[32:])]
    signs=2*encoded.astype(float)-1
    gain=np.mean(soft*signs)
    return {'words':words,'crc_ok':computed==observed,'computed_crc':computed,'observed_crc':observed,
            'zero_tails':bool(all(np.all(b[32:]==0) for b in blocks)),
            'phase_rad':phase,'slope_rad_per_carrier':slope,'polarity':polarity,
            'coded_hard_disagreements':int(np.sum((soft>0).astype(np.uint8)!=encoded)),
            'decision_directed_normalized_residual':float(np.mean(abs(soft-gain*signs)**2)/max(np.mean(abs(soft)**2),1e-30)),
            'prefix_coded_correlation':float(np.dot(soft[:48],reference)/max(np.linalg.norm(soft[:48])*np.linalg.norm(reference),1e-30)),
            'crc_used_for_parameter_selection':False,'soft':soft,'retained_bits':bits,'reencoded':encoded}

def extract_header(iq,frame):
    carriers=header_carriers();start=slot_start(frame,2)
    spectrum=np.fft.fft(iq[start:start+FFT_SIZE].astype(np.complex128))
    values=spectrum[carriers%FFT_SIZE]
    values*=np.array([pn_reference(int(k),np.array([2]))[0].conjugate() for k in carriers])
    return values,carriers
