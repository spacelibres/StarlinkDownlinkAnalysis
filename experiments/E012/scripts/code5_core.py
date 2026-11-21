"""Inherited-model receiver calibration and explicit GF(2) parity recovery."""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));import project_paths
from receiver_models import pn_reference

LOADED=np.r_[np.arange(-512,-2),np.arange(2,512)]
PILOTS=np.r_[np.arange(-496,-488),np.arange(488,496)]
TRAIN_PILOTS=np.arange(0,16,2)
PERIOD=640009.32


def reference_gain(observed,reference):
    observed=np.asarray(observed);reference=np.asarray(reference)
    return np.vdot(reference,observed)/np.vdot(reference,reference)


def sinc_sample(values,points):
    points=np.asarray(points,dtype=float)
    indices=np.floor(points).astype(int)[:,None]+np.arange(-7,9)[None,:]
    if indices.min()<0 or indices.max()>=len(values):raise IndexError('Interpolation stencil outside capture')
    delta=points[:,None]-indices;weights=np.sinc(delta)*np.sinc(delta/8)
    weights[abs(delta)>=8]=0;weights/=weights.sum(axis=1,keepdims=True)
    return np.sum(np.asarray(values[indices],dtype=np.complex128)*weights,axis=1)


def fft_grid(iq,frame,correct_cfo=True):
    spectra=[];scale=PERIOD/640000;center=(len(iq)-1)/2/480e6
    for slot in range(1,302):
        positions=639992+frame*PERIOD+(slot*2112+64+np.arange(2048))*scale
        values=sinc_sample(iq,positions)
        if correct_cfo:
            time=positions/480e6-center
            values*=np.exp(-1j*(2*np.pi*48257.986*time+np.pi*(-1107.157)*time*time))
        spectra.append(np.fft.fft(values)[LOADED%2048])
    return np.array(spectra)


def equalize_grid(received,sss_observed,sss_reference,slots):
    received=np.asarray(received);slots=np.asarray(slots)
    channel=np.asarray(sss_observed)/np.asarray(sss_reference)
    channel=np.r_[gaussian_filter1d(channel[:510],2,mode='nearest',truncate=4),
                  gaussian_filter1d(channel[510:],2,mode='nearest',truncate=4)]
    pidx=np.array([np.flatnonzero(LOADED==k)[0] for k in PILOTS])
    nonpilot=~np.isin(LOADED,PILOTS);pn=np.column_stack([pn_reference(int(k),slots) for k in LOADED])
    equal=received/channel[None,:]
    ratios=equal[:,pidx]/pn[:,pidx]
    k=PILOTS[TRAIN_PILOTS].astype(float);design=np.c_[np.ones(len(k)),k]
    phases=np.unwrap(np.angle(ratios[:,TRAIN_PILOTS]),axis=1)
    coefficients=np.linalg.lstsq(design,phases.T,rcond=None)[0].T
    gain=np.median(abs(ratios[:,TRAIN_PILOTS]),axis=1)
    correction=np.exp(-1j*(coefficients[:,0,None]+coefficients[:,1,None]*LOADED[None,:]))/gain[:,None]
    corrected=equal*correction;reserved=np.setdiff1d(np.arange(16),TRAIN_PILOTS)
    held=corrected[:,pidx[reserved]]/pn[:,pidx[reserved]]
    return {'nonpilot':corrected[:,nonpilot]*pn[:,nonpilot].conjugate(),
            'calibration':np.c_[coefficients,gain],
            'reserved_pilot_evm':float(np.sqrt(np.mean(abs(held-1)**2))),
            'reserved_pilot_label_errors':int(np.sum(abs(np.angle(held))>np.pi/4)),
            'reserved_pilot_count':int(held.size),'reserved_pilot_ratio':held,'channel':channel}


def qc_augment(systematic,parity,lift):
    systematic=np.asarray(systematic,dtype=np.uint8);parity=np.asarray(parity,dtype=np.uint8)
    if systematic.shape[1]%lift or parity.shape[1]%lift:raise ValueError('Unaligned QC blocks')
    def shifted(a):
        shaped=a.reshape(len(a),-1,lift)
        return np.concatenate([np.roll(shaped,j,axis=2).reshape(a.shape) for j in range(lift)])
    return shifted(systematic),shifted(parity)


def recover_generator(systematic,parity):
    u=np.asarray(systematic,dtype=np.uint8);v=np.asarray(parity,dtype=np.uint8)
    if u.ndim!=2 or v.ndim!=2 or len(u)!=len(v):raise ValueError('Equation shape mismatch')
    k=u.shape[1];m=v.shape[1];mask=(1<<k)-1;basis=[0]*k;rank=0;bad=0
    def packed(a):return [int.from_bytes(row.tobytes(),'little') for row in np.packbits(a,axis=1,bitorder='little')]
    for left,right in zip(packed(u),packed(v)):
        row=left|(right<<k)
        while row&mask:
            pivot=((row&mask)&-(row&mask)).bit_length()-1
            if basis[pivot]:row^=basis[pivot]
            else:basis[pivot]=row;rank+=1;break
        else:
            if row>>k:bad+=1
    generator=None
    if rank==k and bad==0:
        solution=[0]*k
        for pivot in range(k-1,-1,-1):
            row=basis[pivot];right=row>>k;left=(row&mask)^(1<<pivot)
            while left:
                bit=left&-left;right^=solution[bit.bit_length()-1];left^=bit
            solution[pivot]=right
        length=(m+7)//8
        generator=np.unpackbits(np.array([list(row.to_bytes(length,'little')) for row in solution],dtype=np.uint8),axis=1,bitorder='little')[:,:m]
    return {'rank':rank,'input_dimensions':k,'free_input_dimensions':k-rank,
            'inconsistent_equations':bad,'generator':generator}
