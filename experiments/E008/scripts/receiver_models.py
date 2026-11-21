"""Numerical receiver checks. Importing this module performs no IQ experiment."""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar

def split_prefix_basis(p,s):
    p=np.asarray(p);s=np.asarray(s)
    u=p/np.linalg.norm(p)
    v=s-u*np.vdot(u,s);v/=np.linalg.norm(v)
    phase=np.vdot(u,p+s)*np.vdot(v,p+s).conj()
    return u,v,np.exp(-1j*np.angle(phase))

@lru_cache(maxsize=1)
def pn_bits():
    bits=np.zeros(32767,dtype=np.uint8)
    bits[:15]=[int(x) for x in '000000100000010']
    for j in range(15,len(bits)):bits[j]=bits[j-14]^bits[j-15]
    return bits

def pn_reference(k,slots):
    signed=int(k) if int(k)<512 else int(k)-1024
    ordinal=signed+512 if signed<0 else signed+508
    d=1020*(np.asarray(slots)-17)+ordinal;bits=pn_bits()
    rotation=(2*bits[(d+16327)%32767].astype(int)+(1^bits[(d+16320)%32767])+1)%4
    return np.exp(.5j*np.pi*rotation)

def cycle_contrast(values,train,test,control_phases,remove_shared=False):
    """Predict class contrasts on reserved whole cycles, removing frame constants.

    values: frame x cycle x 15 classes. The first/last two classes are excluded.
    Control phases are shared across all frames/classes of each reserved cycle.
    """
    values=np.asarray(values)[:,:,2:13]
    a=values[:,train].mean(axis=1);a=a-a.mean(axis=1,keepdims=True)
    percycle=values[:,test].transpose(1,0,2)
    percycle=percycle-percycle.mean(axis=2,keepdims=True)
    if remove_shared:
        # Fit the shared class direction on calibration cycles only. A fixed
        # reference times a changing frame gain must not count as new messages.
        _,_,v=np.linalg.svd(a,full_matrices=False);direction=v[0]
        a-=np.einsum('fj,j->f',a,direction.conj())[:,None]*direction[None,:]
        percycle-=np.einsum('cfj,j->cf',percycle,direction.conj())[:,:,None]*direction[None,None,:]
    q=np.einsum('fj,cfj->c',a.conj(),percycle)
    null=(np.exp(1j*control_phases)@q/len(test)).real
    actual=float(q.mean().real);scale=max(float(null.std(ddof=1)),1e-30)
    b=percycle.mean(axis=0)
    coherence=float(np.real(np.vdot(a,b))/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))
    return {'score':actual/scale,'coherence':coherence,'null_scores':null/scale,
            'training_contrasts':a,'heldout_contrasts':b,'per_cycle_projection':q}

def fit_frame(samples,anchor,frame,expected,training_slots,fs=2500000.,edge_guard_samples=.6,*,withheld_pilot=None):
    """Fit CFO/shared pilot gains only on calibration slots; unknown tones free.

    Time-domain Schur projection avoids attributing nearby unknown tones to known
    pilots. Reserved-cycle errors are predictions, not same-data constellation EVM.
    The finite narrowband/filter model remains approximate on real IQ.
    A withheld pilot is handled as an unknown tone for receiver controls.
    """
    slots=np.arange(2,302);ts=4.4e-6;cp=2/15*1e-6
    epoch=anchor['integer_epoch_sample']+anchor['fractional_epoch_offset_samples']+frame*fs/750
    first=int(np.ceil(epoch+2*ts*fs))
    n=first+np.arange(300)[:,None]*11+np.arange(11)[None,:]
    local=(n-epoch)/fs-slots[:,None]*ts
    keep=np.all((local>=cp+edge_guard_samples/fs)&(local<=ts-edge_guard_samples/fs),axis=0)
    n=n[:,keep];local=local[:,keep]
    if n.min()<0 or n.max()>=len(samples):return None
    cfo=anchor['fractional_tracking_cfo_hz'];indices=np.arange(520,545)
    tones=(indices-531.5)*234375
    allowed=abs(tones+cfo)<fs*.49;indices=indices[allowed];tones=tones[allowed]
    known=(indices>=528)&(indices<=535)
    if withheld_pilot is not None:
        known &= indices != withheld_pilot
    design=np.exp(2j*np.pi*(local-cp)[:,:,None]*tones[None,None,:])/np.sqrt(8)
    p=design[:,:,known]*expected[:,indices[known]-528][:,None,:]
    u=design[:,:,~known];inverse=np.linalg.pinv(u)
    qp=p-np.einsum('snq,sqr,srp->snp',u,inverse,p)
    train=np.asarray(training_slots,dtype=bool);test=(~train)&(np.arange(300)>=30)
    gram=np.einsum('snp,snr->pr',qp[train].conj(),p[train]);inv=np.linalg.pinv(gram)
    coupling=np.einsum('sqn,snp->sqp',inverse,p)
    unit_variance=np.sum(abs(inverse)**2,axis=2)+np.einsum('sqp,pr,sqr->sq',coupling,inv,coupling.conj()).real
    times=n/fs;times-=times[train].mean()
    raw=samples[n]*np.exp(-2j*np.pi*cfo*n/fs)
    def evaluate(df,full=False):
        y=raw*np.exp(-2j*np.pi*df*times)
        h=inv@np.einsum('snp,sn->p',qp[train].conj(),y[train])
        qy=y-np.einsum('snq,sqr,sr->sn',u,inverse,y)
        prediction=np.einsum('snp,p->sn',qp,h)
        error=qy-prediction
        cost=float(np.sum(abs(error[train])**2))
        if not full:return cost
        unknown=np.einsum('sqn,sn->sq',inverse,y-np.einsum('snp,p->sn',p,h))
        return {'indices':indices,'unknown_indices':indices[~known],'unknown':unknown,
                'pilot_gains':h,'residual_cfo_hz':float(df),'fit_at_search_boundary':bool(abs(df)>=800),
                'heldout_projected_normalized_error':float(np.sum(abs(error[test])**2)/max(np.sum(abs(qy[test])**2),1e-30)),
                'calibration_projected_normalized_error':float(np.sum(abs(error[train])**2)/max(np.sum(abs(qy[train])**2),1e-30)),
                'pilot_gram_condition':float(np.linalg.cond(gram)),
                'unknown_design_condition':float(np.median(np.linalg.cond(u))),
                'samples_per_symbol':int(n.shape[1]),
                'heldout_unit_noise_variance_by_unknown':np.median(unit_variance[test],axis=0).tolist()}
    grid=np.arange(-800,801,100);costs=np.array([evaluate(x) for x in grid]);winner=float(grid[costs.argmin()])
    optimum=minimize_scalar(evaluate,bounds=(winner-100,winner+100),method='bounded',options={'xatol':.05})
    return evaluate(optimum.x,True)
