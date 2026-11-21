"""Independent-capture front end; no learned parity/check parameters here."""
from pathlib import Path
import sys
import numpy as np
from scipy import signal

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));import project_paths
from code5_core import LOADED,PILOTS,TRAIN_PILOTS,sinc_sample
from protocol_core import decode_soft_block
from layout_core import prefix_fields,parse_descriptors
from educational_fec_demo import encode_block,crc8
from receiver_models import pn_reference


def read_ci16(path):
    path=Path(path)
    if path.stat().st_size%4:raise ValueError('Incomplete int16 IQ pair')
    raw=np.memmap(path,dtype='<i2',mode='r').reshape(-1,2)
    return raw[:,0].astype(np.float32)+1j*raw[:,1].astype(np.float32)


def resample_480(values):
    coefficients=signal.firwin(6145,1/48,window=('kaiser',8.6))
    return signal.resample_poly(values,48,25,window=coefficients).astype(np.complex64)


def pss_reference():
    reg=[0,0,1,1,0,1,0];bits=[]
    for _ in range(127):
        b=reg[2]^reg[6];reg=[b]+reg[:-1];bits.append(b)
    segment=np.exp(-.25j*np.pi-.5j*np.pi*np.cumsum(2*np.r_[0,bits[::-1]]-1))
    useful=np.r_[-segment,np.tile(segment,7)]
    spectrum=np.fft.fftshift(np.fft.fft(useful));upsampled=2*np.fft.ifft(np.fft.ifftshift(np.pad(spectrum,(512,512))))
    return np.r_[-upsampled[-64:],upsampled]


def repetition_candidates(values):
    delay=256;window=1536;length=len(values)-delay-window+1
    profile=np.empty(length,dtype=np.float32)
    for first in range(0,length,1048576):
        stop=min(length,first+1048576);a=values[first:stop+window-1];b=values[first+delay:stop+window-1+delay]
        def moving(x):
            cumulative=np.r_[0,np.cumsum(x,dtype=np.complex128 if np.iscomplexobj(x) else np.float64)]
            return cumulative[window:]-cumulative[:-window]
        product=moving(b*a.conj());power=moving(abs(a)**2)*moving(abs(b)**2)
        profile[first:stop]=abs(product)/np.sqrt(np.maximum(power,1e-30))
    peaks,_=signal.find_peaks(profile,height=.5,distance=2112)
    candidates=[]
    for peak in peaks:
        product=np.sum(values[peak+delay:peak+delay+window]*values[peak:peak+window].conj())
        candidates.append({'rough_epoch':int(peak)-320,'repetition_score':float(profile[peak]),
                           'coarse_frequency_hz':float(np.angle(product)*480e6/(2*np.pi*delay))})
    return candidates


def refine_pss(values,rough_epoch,coarse_frequency):
    template=pss_reference();first=max(0,int(round(rough_epoch))-128);last=min(len(values),int(round(rough_epoch))+len(template)+129)
    if last-first<len(template):raise IndexError('PSS template not fully observed')
    local=np.asarray(values[first:last]);n=first+np.arange(len(local));best=None
    energy=signal.convolve(abs(local)**2,np.ones(len(template)),mode='valid')
    for alias in range(-2,3):
        frequency=coarse_frequency+alias*480e6/256
        corrected=local*np.exp(-2j*np.pi*frequency*n/480e6)
        correlation=signal.correlate(corrected,template,mode='valid',method='fft')
        scores=abs(correlation)/np.sqrt(np.maximum(energy*np.sum(abs(template)**2),1e-30))
        peak=int(np.argmax(scores));fraction=0.
        if 0<peak<len(scores)-1:
            a,b,c=scores[peak-1:peak+2]**2;denominator=a-2*b+c
            if denominator:fraction=float(np.clip(.5*(a-c)/denominator,-.5,.5))
        row={'epoch':first+peak+fraction,'score':float(scores[peak]),'cfo_hz':float(frequency),'frequency_alias':alias}
        if best is None or row['score']>best['score']:best=row
    return best


def pilot_residual(ratios,slots):
    slots=np.asarray(slots);use=(slots>=50)&(slots<=289)
    if np.sum(use)<32:raise ValueError('Insufficient training slots for clock/frequency')
    time=slots[use]*4.4e-6;design=np.c_[np.ones(len(time)),time]
    phase=np.unwrap(np.angle(np.asarray(ratios)[use][:,TRAIN_PILOTS]),axis=0)
    slopes=np.linalg.lstsq(design,phase,rcond=None)[0][1]
    frequency_slope=np.linalg.lstsq(np.c_[np.ones(8),PILOTS[TRAIN_PILOTS]],slopes,rcond=None)[0]
    return {'frequency_hz':float(frequency_slope[0]/(2*np.pi)),
            'clock_fraction':float(-frequency_slope[1]/(2*np.pi*234375)),'training_slots':int(np.sum(use))}


def capture_grid(values,epoch,scale,frequency):
    rows=[];slots=[]
    for slot in range(1,302):
        positions=epoch+(slot*2112+64+np.arange(2048))*scale
        if positions.min()<8 or positions.max()+9>=len(values):break
        data=sinc_sample(values,positions)*np.exp(-2j*np.pi*frequency*positions/480e6)
        rows.append(np.fft.fft(data)[LOADED%2048]);slots.append(slot)
    if len(rows)<2:raise IndexError('No complete SSS and first data slot')
    return np.array(rows),np.array(slots)


def pilot_ratios(spectra,slots,sss_reference):
    channel=spectra[0]/sss_reference;pidx=np.array([np.flatnonzero(LOADED==k)[0] for k in PILOTS])
    data=spectra[1:,pidx]/channel[pidx]
    expected=np.column_stack([pn_reference(int(k),slots[1:]) for k in PILOTS])
    return data/expected


def decode_variable_pdu(normalized,kind):
    normalized=np.asarray(normalized);first=decode_soft_block(-normalized[:114].real);fields=prefix_fields(first[:32])
    count=fields['blocks'];entries=fields['entry_count']
    admitted=(2<=count<=7 and 1<=entries<=15 and 16+20*entries+8<=32*count)
    admitted=admitted and ((kind=='h1' and fields['prefix']==0x3048 and count==2) or
                          (kind=='h2' and fields['prefix']>>8==0x34 and fields['second_header_role']==1))
    result={**fields,'precrc_admitted':bool(admitted),'crc_ok':None,'crc_used_for_selection':False}
    if not admitted:return result
    needed=114*count;result['needed_resources']=needed
    if len(normalized)<needed:return {**result,'insufficient_resources':True}
    bits=np.concatenate([decode_soft_block(-normalized[j:j+114].real)[:32] for j in range(0,needed,114)])
    coded=np.concatenate([encode_block(bits[j:j+32]) for j in range(0,len(bits),32)])
    observed=int(''.join(map(str,bits[-8:])),2)
    result.update(bits=bits,reencoded=coded,crc_ok=crc8(bits[:-8])==observed,
                  coded_hard_disagreements=int(np.sum((normalized[:needed].real<0)!=coded)),
                  words_hex=[f'0x{int("".join(map(str,bits[j:j+32])),2):08X}' for j in range(0,len(bits),32)],
                  layout20=parse_descriptors(bits,20))
    return result
