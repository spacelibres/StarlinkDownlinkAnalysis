"""Narrowband known-pilot demodulation, with no spacecraft-ID interpretation."""

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

ROOT=LEO_SNAPSHOT;OUT=experiment_output('E003', '')
source=ROOT/'reports/2026_10_18_dsx01_pilot_clock/DSX01-S05'
# Read only the factual, published pilot integers; avoid importing acquisition infrastructure.
tree=ast.parse((ROOT/'src/leo/analysis/starlink/templates.py').read_text())
constants=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QIN_EDGE_PILOT_HEX_V1' for t in n.targets))
states=np.array([[(int(constants[k],16)>>(2*(299-j)))&3 for k in range(528,536)] for j in range(300)])
expected=np.exp(.5j*np.pi*(states+.5));control=np.roll(expected,17,axis=0)
meta=json.loads((source/'selection.json').read_text());fs=meta['sample_rate_hz'];symbol_s=4.4e-6;cp_s=2/15*1e-6
symbols=np.arange(2,302);starts=np.rint(symbols*fs*symbol_s).astype(int);assert np.all(np.diff(starts)==11)
relative=starts[:,None]+np.arange(11)
tones=(np.arange(8)-3.5)*234375
local=relative/fs-symbols[:,None]*symbol_s-cp_s
basis=np.exp(2j*np.pi*local[:,:,None]*tones[None,None,:])/np.sqrt(8)
solve=np.linalg.pinv(basis)
offsets=np.arange(-7,9)
def interpolate(samples,positions):
    inds=np.floor(positions).astype(int)[...,None]+offsets
    assert inds.min()>=0 and inds.max()<len(samples)
    delta=positions[...,None]-inds
    w=np.sinc(delta)*np.sinc(delta/8);w/=w.sum(axis=-1,keepdims=True)
    return np.sum(samples[inds]*w,axis=-1)

def demod(samples,epoch,fraction,cfo):
    frame_starts=[]
    for j in range(16):
        start=epoch+round(j*fs/750)
        if start+fraction+round(302*fs*symbol_s)-1>=len(samples)-8:break
        if start+fraction+starts[0]>=7:frame_starts.append(start)
    positions=np.array(frame_starts)[:,None,None]+relative[None,:,:]+fraction
    values=interpolate(samples,positions)*np.exp(-2j*np.pi*cfo*positions/fs)
    return np.einsum('stn,fsn->fst',solve,values),frame_starts

def score(cube,template):
    residuals=np.arange(-1000,1001,20)
    times=(np.arange(300)+2.5)*symbol_s;times-=times.mean()
    phasors=np.exp(-2j*np.pi*residuals[:,None]*times)
    matched=cube*np.conj(template)[None,:,:]
    correlations=np.einsum('ds,fst->dft',phasors,matched,optimize=True)
    scores=np.sum(abs(correlations)**2,axis=(1,2))/(300*np.sum(abs(cube)**2))
    k=int(scores.argmax());return float(scores[k]),float(residuals[k]),phasors[k]

results=[];arrays={}
with np.load(source/'selected_iq.npz',allow_pickle=False) as z:
    for window in meta['windows']:
        key='iq_'+str(window['number']);anchor=window['winner'];samples=z[key]
        cube,frames=demod(samples,anchor['integer_epoch_sample'],anchor['fractional_epoch_offset_samples'],anchor['fractional_tracking_cfo_hz'])
        exact,df,phase=score(cube,expected);null,null_df,_=score(cube,control)
        corrected=cube*phase[None,:,None]
        channels=np.mean(corrected*np.conj(expected)[None,:,:],axis=1)
        equalized=corrected/np.where(abs(channels[:,None,:])>1e-20,channels[:,None,:],1)
        hard=np.rint((np.angle(equalized)-np.pi/4)/(np.pi/2)).astype(int)%4
        agreement=float(np.mean(hard==states[None,:,:]))
        row={'array_key':key,'track':window['track'],'frames':len(frames),'known_states_per_frame':2400,'exact_score':exact,'rolled_control_score':null,'margin':exact-null,'residual_frequency_hz':df,'descriptive_known_symbol_agreement':agreement,'satellite_identity_decoded':False,'note':'Per-frame channels fitted using known pilots; symbol agreement is descriptive, not a calibrated unknown-payload error rate. Acquisition anchors inherited from selection.json.'}
        results.append(row);arrays[key+'_pilot_cube']=cube.astype(np.complex64);arrays[key+'_equalized']=equalized.astype(np.complex64)
        print(key,'frames',len(frames),'exact',round(exact,4),'control',round(null,4),'agreement',round(agreement,4),flush=True)
arrays['expected_states']=states.astype(np.int8)
np.savez_compressed(artifact_path('decoded_known_pilot_symbols.npz'),**arrays)
(artifact_path('known_pilot_verification.json')).write_text(json.dumps({'method':'Known lower-edge pilot demodulation and residual-frequency correlation; universal template has no per-spacecraft identifier.','identity_decoded':False,'windows':results},indent=2),encoding='utf8')
assert len(results)==8 and all(r['margin']>.05 for r in results)
print('PASS: all eight retained windows match the known pilot better than the rolled control; no ID field was decoded.')
