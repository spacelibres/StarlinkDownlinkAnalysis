"""Check whether the nonpilot structure survives a known-pilot-only null signal."""

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
import ast,json
import numpy as np
from scipy.io import loadmat

OUT=experiment_output('E003', '')
SRC=LEO_SNAPSHOT
source=SRC/'reports/2026_10_18_dsx01_pilot_clock/DSX01-S05'
meta=json.loads((source/'selection.json').read_text());fs=meta['sample_rate_hz']
tree=ast.parse((SRC/'src/leo/analysis/starlink/templates.py').read_text())
const=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QIN_EDGE_PILOT_HEX_V1' for t in n.targets))
expected=np.exp(.5j*np.pi*(np.array([[(int(const[k],16)>>(2*(299-j)))&3 for k in range(528,536)] for j in range(300)])+.5))
rot=loadmat(artifact_path('reference/qin/referenceTemplate.mat'))['referenceTemplateRotations']
symbol=np.arange(2,302);relative=symbol[:,None]*11+np.arange(11);local=relative/fs-symbol[:,None]*4.4e-6-2/15*1e-6
cl=(symbol-2)%15;cy=(symbol-2)//15
perm=np.random.default_rng(20261120).permutation(np.arange(2,20));train=perm[:9];test=perm[9:]
def repetition(y,reference):
    d=y*reference.conj()[None,:]
    a=np.array([d[:,(cl==j)&np.isin(cy,train)].mean(axis=1) for j in range(2,13)]).T
    b=np.array([d[:,(cl==j)&np.isin(cy,test)].mean(axis=1) for j in range(2,13)]).T
    return float(np.real(np.vdot(a,b))/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real))
def demod(samples,anchor):
    indices=np.arange(527,537);f=(indices-531.5)*234375;cfo=anchor['fractional_tracking_cfo_hz'];keep=abs(f+cfo)<fs/2;indices=indices[keep];f=f[keep]
    design=np.exp(2j*np.pi*local[:,:,None]*f[None,None,:])/np.sqrt(8);solve=np.linalg.pinv(design)
    starts=[]
    for j in range(16):
        start=anchor['integer_epoch_sample']+round(j*fs/750)
        if start+anchor['fractional_epoch_offset_samples']+3321>=len(samples)-8:break
        starts.append(start)
    pos=np.array(starts)[:,None,None]+relative[None,:,:]+anchor['fractional_epoch_offset_samples']
    idx=np.floor(pos).astype(int)[...,None]+np.arange(-7,9);delta=pos[...,None]-idx
    weights=np.sinc(delta)*np.sinc(delta/8);weights/=weights.sum(axis=-1,keepdims=True)
    values=np.sum(samples[idx]*weights,axis=-1)*np.exp(-2j*np.pi*cfo*pos/fs)
    return np.einsum('stn,fsn->fst',solve,values),indices
def synth(anchor,seed=20261120):
    rng=np.random.default_rng(seed);n=np.arange(50000);epoch=anchor['integer_epoch_sample']+anchor['fractional_epoch_offset_samples']
    t=(n-epoch)/fs;within=np.mod(t,1/750);i=np.floor(within/4.4e-6).astype(int)
    active=(i>=2)&(i<302);out=np.zeros(len(n),complex)
    tones=(np.arange(8)-3.5)*234375
    gains=np.exp(1j*rng.uniform(-np.pi,np.pi,8))
    for k in range(8):
        out[active]+=gains[k]*expected[i[active]-2,k]*np.exp(2j*np.pi*tones[k]*(within[active]-i[active]*4.4e-6-2/15*1e-6))/np.sqrt(8)
    out*=np.exp(2j*np.pi*anchor['fractional_tracking_cfo_hz']*n/fs)
    return out

def main():
    rows=[]
    for w in meta['windows'][:4]:
        a=w['winner'];s=synth(a)
        for mode,noise in [('pilot_only_noiseless',0),('pilot_only_noise_variance_5',5)]:
            rng=np.random.default_rng(20261121+w['number']);samples=s+np.sqrt(noise/2)*(rng.normal(size=len(s))+1j*rng.normal(size=len(s)))
            cube,indices=demod(samples,a)
            for k in [527,536]:
                if k not in indices:continue
                values=cube[:,:,np.where(indices==k)[0][0]];reference=np.exp(.5j*np.pi*rot[k,symbol-1])
                p=float(np.mean(abs(values)**2));score=repetition(values,reference)
                rows.append({'window':w['number'],'mode':mode,'subcarrier':k,'estimated_nonpilot_power':p,'repetition_score':score,'note':'No nonpilot information was transmitted in this synthetic control.'})
                print(w['number'],mode,k,'power',round(p,6),'repetition',round(score,4),flush=True)
    (artifact_path('nonpilot_leakage_audit.json')).write_text(json.dumps({'controls':rows},indent=2),encoding='utf8')

if __name__=='__main__':main()
