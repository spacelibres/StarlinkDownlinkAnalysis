"""Resolve available nonpilot T-code positions from a simultaneous two-RX window.

Only known pilot/SSS symbols calibrate phase. Unknown data are never used to choose
satellite identities. Output codewords retain erasures and are not message fields.
"""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E003')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv,json
import numpy as np
from scipy import fft
from scipy.io import loadmat
import audit_nonpilot_leakage as core

OUT=experiment_output('E003', '')
PAIR=core.source.parent/'paired_RX'
meta=json.loads((PAIR/'selection.json').read_text());fs=meta['sample_rate_hz'];symbol=np.arange(1,302)
relative=symbol[:,None]*11+np.arange(11)
local=relative/fs-symbol[:,None]*4.4e-6-2/15*1e-6
times=(symbol+.5)*4.4e-6;times-=np.mean(times[1:])
sss=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel()
def demod(samples,anchor):
    cfo=anchor['fractional_tracking_cfo_hz']
    indices=np.arange(520,545);tones=(indices-531.5)*234375
    keep=abs(tones+cfo)<fs*.49;indices=indices[keep];tones=tones[keep]
    design=np.exp(2j*np.pi*local[:,:,None]*tones[None,None,:])/np.sqrt(8)
    solve=np.linalg.pinv(design)
    starts=[]
    for j in range(16):
        start=anchor['integer_epoch_sample']+round(j*fs/750)
        if start+anchor['fractional_epoch_offset_samples']+3321>=len(samples)-8:break
        starts.append(start)
    pos=np.array(starts)[:,None,None]+relative[None,:,:]+anchor['fractional_epoch_offset_samples']
    idx=np.floor(pos).astype(int)[...,None]+np.arange(-7,9);d=pos[...,None]-idx
    w=np.sinc(d)*np.sinc(d/8);w/=w.sum(axis=-1,keepdims=True)
    sampled=np.sum(samples[idx]*w,axis=-1)*np.exp(-2j*np.pi*cfo*pos/fs)
    cube=np.einsum('stn,fsn->fst',solve,sampled)
    ispilot=np.isin(indices,np.arange(528,536));pilots=cube[:,1:,ispilot]
    exp=core.expected[:,indices[ispilot]-528]
    matched=pilots*exp.conj()[None,:,:];size=16384;f=fft.fftfreq(size,4.4e-6);bins=np.where(abs(f)<2000)[0]
    trans=fft.fft(matched,n=size,axis=1);power=np.sum(abs(trans)**2,axis=-1);winner=bins[power[:,bins].argmax(axis=1)]
    x0=power[np.arange(len(cube)),(winner-1)%size];x1=power[np.arange(len(cube)),winner];x2=power[np.arange(len(cube)),(winner+1)%size]
    delta=np.clip(.5*(x0-x2)/np.minimum(x0-2*x1+x2,-1e-20),-.5,.5)
    df=f[winner]+delta/(size*4.4e-6)
    cube*=np.exp(-2j*np.pi*df[:,None,None]*times[None,:,None])
    h=np.mean(cube[:,1:,ispilot]*exp.conj()[None,:,:],axis=1)
    # Smooth frequency-phase model, trained exclusively on known pilots.
    x=indices[ispilot]-531.5
    amp=np.maximum(abs(h),1e-12);phase=np.unwrap(np.angle(h),axis=1)
    prediction=[];held_errors=[]
    for j in range(len(cube)):
        coef=np.polyfit(x,phase[j],1,w=amp[j]);gain=np.polyfit(x,np.log(amp[j]),1,w=amp[j])
        prediction.append(np.exp(np.polyval(gain,indices-531.5)+1j*np.polyval(coef,indices-531.5)))
        errs=[]
        for k in range(len(x)):
            use=np.arange(len(x))!=k;c=np.polyfit(x[use],phase[j,use],1,w=amp[j,use]);errs.append(np.angle(np.exp(1j*(phase[j,k]-np.polyval(c,x[k])))))
        held_errors.append(np.median(abs(np.array(errs))))
    prediction=np.array(prediction)
    sss_ratios=cube[:,0,:]/(prediction*sss[indices][None,:])
    # Leave each frame out of the SSS calibration used for its data decisions.
    ratio=np.array([np.mean(np.delete(sss_ratios,j,axis=0),axis=0) for j in range(len(cube))])
    sss_noise=np.std(sss_ratios,axis=0,ddof=1)/np.sqrt(max(len(cube)-1,1))
    calibration_snr=abs(np.mean(sss_ratios,axis=0))/np.maximum(sss_noise,1e-12)
    equalized=cube/(prediction[:,None,:]*ratio[:,None,:])
    return {'indices':indices,'cube':cube,'equalized':equalized,'frame_cfo':df,'sss_ratios':sss_ratios,'sss_calibration_snr':calibration_snr,'pilot_phase_prediction_median_abs_error_rad':np.array(held_errors)}

decoded={};raw={}
with np.load(PAIR/'selected_iq.npz',allow_pickle=False) as z:
    for number in [2,3]:
        w=meta['windows'][number];key='iq_'+str(number);decoded[key]=demod(z[key],w['winner']);raw[key]=z[key]
        d=decoded[key];print(key,'observed carriers',d['indices'].tolist(),'SSS SNR',np.round(d['sss_calibration_snr'],2).tolist(),'phase prediction error deg',round(float(np.median(d['pilot_phase_prediction_median_abs_error_rad']))*180/np.pi,1),flush=True)

features=[];stats=[];arrays={};frames=min(len(d['cube']) for d in decoded.values())
num=np.zeros((frames,60));den=np.zeros((frames,60));counts=np.zeros((frames,60),int)
individual={}
for key,d in decoded.items():
    indices=d['indices'];arrays[key+'_indices']=indices;arrays[key+'_equalized']=d['equalized'].astype(np.complex64);arrays[key+'_sss_ratios']=d['sss_ratios'].astype(np.complex64)
    for c,k in enumerate(indices):
        if 528<=k<=535:continue
        reference=np.exp(.5j*np.pi*core.rot[k,symbol-1]);deviation=d['equalized'][:,:,c]*reference.conj()[None,:]
        means=np.full((frames,60),np.nan+1j*np.nan);variances=np.full((frames,60),np.nan)
        for bit in range(60):
            use=(symbol>31)&(((int(k)-16*symbol)%60)==bit)
            if not np.any(use):continue
            values=deviation[:frames,use];m=values.mean(axis=1);variance=np.var(values,axis=1,ddof=1)/values.shape[1]/2
            means[:,bit]=m;variances[:,bit]=variance
            # Calibration must resolve phase independently of unknown data.
            good=bool(d['sss_calibration_snr'][c]>3)
            if good:
                weight=1/np.maximum(variance,1e-12);num[:,bit]+=weight*m.real;den[:,bit]+=weight;counts[:,bit]+=1
        individual[(key,int(k))]=(means,variances)
        stats.append({'array':key,'subcarrier':int(k),'covered_tcode_positions':np.flatnonzero(np.any(np.isfinite(means),axis=0)).tolist(),'sss_calibration_snr':float(d['sss_calibration_snr'][c]),'phase_calibration_pass':bool(d['sss_calibration_snr'][c]>3)})
        print(key,'nonpilot',int(k),'SSS calibration support',round(float(d['sss_calibration_snr'][c]),2),'positions',int(np.any(np.isfinite(means),axis=0).sum()),flush=True)

soft=num/np.maximum(den,1e-12);stderr=1/np.sqrt(np.maximum(den,1e-12));good=(den>0)&(abs(soft)>3*stderr)
strings=[''.join('1' if soft[j,k]>0 else '0' if soft[j,k]<0 else '?' for k in range(60)) for j in range(frames)]
masked=[''.join(strings[j][k] if good[j,k] else '?' for k in range(60)) for j in range(frames)]
# Same T-code positions observed on different RXs permit a direct redundancy check.
left,lv=individual.get(('iq_2',537),(None,None));right,rv=individual.get(('iq_3',525),(None,None))
overlap={}
if left is not None and right is not None:
    m=np.isfinite(left)&np.isfinite(right);a=left[m];b=right[m]
    overlap={'overlapping_observations':int(m.sum()),'complex_correlation':float(abs(np.vdot(a,b))/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real)),'signed_real_correlation':float(np.real(np.vdot(a,b))/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real)),'note':'Uses leave-one-frame-out SSS phase calibration; exploratory, no decoded-message claim.'}
result={'source':'paired_RX/selected_iq.npz iq_2 and iq_3','simultaneous_source_context_time_s':meta['windows'][2]['context_time_s'],'frames':frames,'subcarriers':stats,'potential_tcode_coverage':sorted({b for r in stats for b in r['covered_tcode_positions']}),'calibrated_covered_positions':int(np.any(den>0,axis=0).sum()),'screened_relative_decisions':int(good.sum()),'candidate_codewords':masked,'complete_screened_codewords':sum('?' not in x for x in masked),'cross_receiver_overlap':overlap,'validated_messages':0,'semantic_fields':{},'warning':'Code candidates require independent consistency and message-level validation. SSS phase support threshold and Gaussian standard errors are screens, not calibrated bit-error probabilities.'}
np.savez_compressed(artifact_path('paired_tcode_observations.npz'),**arrays,soft=soft,stderr=stderr,supported=good,source_count=counts)
(artifact_path('paired_tcode_results.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
with (artifact_path('paired_tcode_candidates.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['frame_index','tcode_candidate_60_positions','validated_message']);w.writerows((j,s,False) for j,s in enumerate(masked))
print('Potential positions',len(result['potential_tcode_coverage']),'calibrated',result['calibrated_covered_positions'],'screened decisions',result['screened_relative_decisions'],'complete words',result['complete_screened_codewords'],flush=True)
print('Cross-RX overlap',overlap,flush=True)
