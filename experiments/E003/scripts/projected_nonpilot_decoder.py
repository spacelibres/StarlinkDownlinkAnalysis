"""Raw-sample frame fit with shared known-pilot gains and per-symbol unknown tones."""

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
from scipy.optimize import minimize_scalar
import audit_nonpilot_leakage as core

OUT=experiment_output('E003', '')
symbol=np.arange(2,302);fs=core.fs;CP=2/15*1e-6;TS=4.4e-6
def frame_decode(samples,anchor,frame):
    epoch=anchor['integer_epoch_sample']+anchor['fractional_epoch_offset_samples']+frame*fs/750
    first=int(np.ceil(epoch+22));n=first+np.arange(300)[:,None]*11+np.arange(11)[None,:]
    if n.max()>=len(samples):return None
    local=(n-epoch)/fs-symbol[:,None]*TS
    # The fractional epoch is used in the continuous basis, not to resample IQ.
    keep=np.all((local>=CP+.6/fs)&(local<=TS-.6/fs),axis=0)
    n=n[:,keep];local=local[:,keep]
    cfo=anchor['fractional_tracking_cfo_hz'];indices=np.arange(520,545);tones=(indices-531.5)*234375
    valid=abs(tones+cfo)<fs*.49;indices=indices[valid];tones=tones[valid]
    known=np.isin(indices,np.arange(528,536));ki=indices[known];ui=indices[~known]
    basis=np.exp(2j*np.pi*(local-CP)[:,:,None]*tones[None,None,:])/np.sqrt(8)
    P=basis[:,:,known]*core.expected[:,ki-528][:,None,:]
    U=basis[:,:,~known];Uplus=np.linalg.pinv(U)
    QP=P-np.einsum('snq,sqr,srp->snp',U,Uplus,P)
    gram=np.einsum('snp,snr->pr',QP.conj(),P)
    inv=np.linalg.inv(gram)
    t=n/fs;center=t.mean();time=t-center
    raw=samples[n]*np.exp(-2j*np.pi*cfo*t)
    def evaluate(df,full=False):
        y=raw*np.exp(-2j*np.pi*df*time)
        rhs=np.einsum('snp,sn->p',QP.conj(),y);h=inv@rhs
        residual=y-np.einsum('snp,p->sn',P,h)
        x=np.einsum('sqn,sn->sq',Uplus,residual)
        error=residual-np.einsum('snq,sq->sn',U,x)
        cost=float(np.sum(abs(error)**2))
        if full:return x,h,cost
        return cost
    grid=np.arange(-800,801,100);cost=np.array([evaluate(df) for df in grid]);seed=grid[cost.argmin()]
    fit=minimize_scalar(evaluate,bounds=(seed-100,seed+100),method='bounded',options={'xatol':.2})
    x,h,error=evaluate(fit.x,full=True)
    phase=np.unwrap(np.angle(h));phasefit=np.polyfit(ki-531.5,phase,1,w=abs(h))
    ampfit=np.polyfit(ki-531.5,np.log(np.maximum(abs(h),1e-12)),1,w=abs(h))
    prediction=np.exp(np.polyval(ampfit,ui-531.5)+1j*np.polyval(phasefit,ui-531.5))
    return {'unknown':ui,'known':ki,'values':x,'pilot_gains':h,'pilot_phase_prediction':prediction,'residual_cfo_hz':float(fit.x),'residual_energy':error,'samples_used_per_symbol':int(n.shape[1]),'rank':int(np.linalg.matrix_rank(gram))}

def structure(values,k):
    ref=np.exp(.5j*np.pi*core.rot[k,core.symbol-1]);return core.repetition(values,ref)

arrays={};results=[];bitrows=[]
with np.load(core.source/'selected_iq.npz',allow_pickle=False) as z:
    for w in core.meta['windows']:
        key='iq_'+str(w['number']);frames=[]
        for frame in range(16):
            f=frame_decode(z[key],w['winner'],frame)
            if f is None:break
            frames.append(f)
        indices=frames[0]['unknown'];cube=np.array([f['values'] for f in frames]);pred=np.array([f['pilot_phase_prediction'] for f in frames])
        arrays[key+'_unknown_indices']=indices;arrays[key+'_values']=cube.astype(np.complex64);arrays[key+'_pilot_prediction']=pred.astype(np.complex64);arrays[key+'_cfo_hz']=np.array([f['residual_cfo_hz'] for f in frames])
        for col,k in enumerate(indices):
            y=cube[:,:,col];ref=np.exp(.5j*np.pi*core.rot[k,core.symbol-1]);score=structure(y,int(k));controls=[core.repetition(y,np.roll(ref,j)) for j in [1,7,19,43,71,113,149]]
            dev=y*ref.conj()[None,:];a=[];b=[];va=[];vb=[]
            for j in range(2,13):
                aa=dev[:,(core.cl==j)&np.isin(core.cy,core.train)];bb=dev[:,(core.cl==j)&np.isin(core.cy,core.test)]
                a.append(aa.mean(axis=1));b.append(bb.mean(axis=1));va.append(np.var(aa,axis=1,ddof=1)/aa.shape[1]/2);vb.append(np.var(bb,axis=1,ddof=1)/bb.shape[1]/2)
            a=np.array(a).T;b=np.array(b).T;sa=np.sqrt(np.array(va).T);sb=np.sqrt(np.array(vb).T)
            theta=.5*np.angle(np.sum(a*a,axis=1));ar=(a*np.exp(-1j*theta[:,None])).real;br=(b*np.exp(-1j*theta[:,None])).real
            mask=(abs(ar)>3*sa)&(abs(br)>3*sb)&(ar*br>0)
            for f in range(len(frames)):
                for j in range(11):bitrows.append({'array':key,'subcarrier':int(k),'frame':f,'symbol_class_mod15':j+2,'relative_sign':('+' if ar[f,j]>0 else '-') if mask[f,j] else '?','support':bool(mask[f,j]),'global_polarity_unknown':True,'message_validated':False})
            row={'array':key,'subcarrier':int(k),'frames':len(frames),'shared_pilot_channel_model':True,'raw_samples_not_interpolated':True,'repetition_score':score,'shifted_control_max':max(controls),'screened_relative_signs':int(mask.sum()),'message_decoded':False}
            results.append(row);print(key,int(k),'repeat',round(score,3),'null',round(max(controls),3),'supported signs',mask.sum(),flush=True)
np.savez_compressed(artifact_path('projected_nonpilot_observations.npz'),**arrays)
(artifact_path('projected_nonpilot_results.json')).write_text(json.dumps({'method':'Raw-sample Schur projection: eight known pilot gains shared across the frame; per-symbol unknown subcarriers remain free; carrier frequency refined only through the projected known-pilot fit. Unknown bits use original grouped cycle split.','results':results,'validated_messages':0},indent=2),encoding='utf8')
with (artifact_path('projected_relative_signs.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(bitrows[0]));writer.writeheader();writer.writerows(bitrows)
print('Done: screened relative signs',sum(r['screened_relative_signs'] for r in results),'complete messages 0',flush=True)
