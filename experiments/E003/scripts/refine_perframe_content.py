"""Estimate carrier correction independently per frame from known pilots only."""

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
import json
import numpy as np
from scipy import fft
import audit_nonpilot_leakage as core

OUT=experiment_output('E003', '')
times=(core.symbol+.5)*4.4e-6;times-=times.mean()
NFFT=16384
freq=fft.fftfreq(NFFT,4.4e-6)
bins=np.where(abs(freq)<=2000)[0]
def frame_correct(cube,indices):
    pilot=cube[:,:,np.isin(indices,np.arange(528,536))]
    matched=pilot*core.expected.conj()[None,:,:]
    transformed=fft.fft(matched,n=NFFT,axis=1)
    powers=np.sum(abs(transformed[:,bins,:])**2,axis=-1)
    winners=bins[powers.argmax(axis=1)]
    allpowers=np.sum(abs(transformed)**2,axis=-1)
    y0=allpowers[np.arange(len(cube)),(winners-1)%NFFT];y1=allpowers[np.arange(len(cube)),winners];y2=allpowers[np.arange(len(cube)),(winners+1)%NFFT]
    delta=.5*(y0-y2)/np.minimum(y0-2*y1+y2,-1e-20)
    residual=freq[winners]+np.clip(delta,-.5,.5)/(NFFT*4.4e-6)
    corrected=cube*np.exp(-2j*np.pi*residual[:,None,None]*times[None,:,None])
    return corrected,residual

rows=[];arrays={}
with np.load(core.source/'selected_iq.npz',allow_pickle=False) as z:
    for w in core.meta['windows'][:4]:
        key='iq_'+str(w['number']);cube,indices=core.demod(z[key],w['winner']);corrected,residual=frame_correct(cube,indices)
        arrays[key+'_indices']=indices;arrays[key+'_cube']=corrected.astype(np.complex64);arrays[key+'_frame_cfo_hz']=residual
        for k in [527,536]:
            if k not in indices:continue
            y=corrected[:,:,np.where(indices==k)[0][0]];ref=np.exp(.5j*np.pi*core.rot[k,core.symbol-1])
            score=core.repetition(y,ref);null=[core.repetition(y,np.roll(ref,j)) for j in [1,7,19,43,71,113,149]]
            row={'window':key,'subcarrier':k,'per_frame_residual_hz':residual.tolist(),'repetition_score':score,'maximum_shifted_reference_control':max(null),'candidate':bool(score>.2 and score>max(null)+.1),'semantic_message_decoded':False}
            rows.append(row);print(key,k,'repeat',round(score,4),'control',round(max(null),4),'CFO range',np.round([residual.min(),residual.max()],1).tolist(),flush=True)
np.savez_compressed(artifact_path('perframe_nonpilot_content.npz'),**arrays)
(artifact_path('perframe_content_results.json')).write_text(json.dumps({'method':'CFO refined separately in each frame using only known pilots; unknown subcarriers not used to select frequency. Repetition check retains original grouped split.','results':rows},indent=2),encoding='utf8')
