"""Exploratory nonpilot-subcarrier test; no payload or field interpretation."""

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

ROOT=LEO_SNAPSHOT;OUT=experiment_output('E003', '')
SOURCE=ROOT/'reports/2026_10_18_dsx01_pilot_clock/DSX01-S05'
meta=json.loads((SOURCE/'selection.json').read_text());fs=meta['sample_rate_hz']
tree=ast.parse((ROOT/'src/leo/analysis/starlink/templates.py').read_text())
constants=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='QIN_EDGE_PILOT_HEX_V1' for t in n.targets))
expected=np.exp(.5j*np.pi*(np.array([[(int(constants[k],16)>>(2*(299-j)))&3 for k in range(528,536)] for j in range(300)])+.5))
rotations=loadmat(artifact_path('reference/qin/referenceTemplate.mat'))['referenceTemplateRotations']
symbol=np.arange(2,302);relative=symbol[:,None]*11+np.arange(11)
times=(symbol+.5)*4.4e-6;times-=times.mean()
offset=np.arange(-7,9)
def interpolate(samples,positions):
    idx=np.floor(positions).astype(int)[...,None]+offset
    delta=positions[...,None]-idx;weights=np.sinc(delta)*np.sinc(delta/8);weights/=weights.sum(axis=-1,keepdims=True)
    assert idx.min()>=0 and idx.max()<len(samples)
    return np.sum(samples[idx]*weights,axis=-1)

cycle=(symbol-2)//15;classes=(symbol-2)%15
eligible_cycles=np.arange(2,20);permutation=np.random.default_rng(20261120).permutation(eligible_cycles)
train_cycles=permutation[:9];test_cycles=permutation[9:]
def repetition_metric(values,ref):
    d=values*ref[None,:].conj();a=[];b=[]
    for j in range(2,13):
        a.append(d[:,(classes==j)&np.isin(cycle,train_cycles)].mean(axis=1))
        b.append(d[:,(classes==j)&np.isin(cycle,test_cycles)].mean(axis=1))
    a=np.array(a).T;b=np.array(b).T
    return float(np.real(np.vdot(a,b))/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real))

results=[];arrays={}
with np.load(SOURCE/'selected_iq.npz',allow_pickle=False) as z:
    for w in meta['windows'][:4]:
        key='iq_'+str(w['number']);anchor=w['winner'];cfo=anchor['fractional_tracking_cfo_hz'];samples=z[key]
        indices=np.arange(527,537);freq=(indices-531.5)*234375
        # Avoid modeling a tone that is physically outside the recorded Nyquist band.
        keep=abs(freq+cfo)<fs/2;indices=indices[keep];freq=freq[keep]
        local=relative/fs-symbol[:,None]*4.4e-6-2/15*1e-6
        design=np.exp(2j*np.pi*local[:,:,None]*freq[None,None,:])/np.sqrt(8)
        solve=np.linalg.pinv(design)
        starts=[]
        for j in range(16):
            start=anchor['integer_epoch_sample']+round(j*fs/750)
            if start+anchor['fractional_epoch_offset_samples']+3321>=len(samples)-8:break
            starts.append(start)
        positions=np.array(starts)[:,None,None]+relative[None,:,:]+anchor['fractional_epoch_offset_samples']
        values=interpolate(samples,positions)*np.exp(-2j*np.pi*cfo*positions/fs)
        cube=np.einsum('stn,fsn->fst',solve,values)
        pilot=cube[:,:,np.isin(indices,np.arange(528,536))]
        grid=np.arange(-1000,1001,20);phasors=np.exp(-2j*np.pi*grid[:,None]*times)
        corr=np.einsum('ds,fst->dft',phasors,pilot*expected.conj()[None,:,:],optimize=True)
        best=int(np.sum(abs(corr)**2,axis=(1,2)).argmax());cube*=phasors[best][None,:,None]
        for index in indices:
            if index in range(528,536):continue
            y=cube[:,:,np.where(indices==index)[0][0]];ref=np.exp(.5j*np.pi*rotations[index,symbol-1])
            score=repetition_metric(y,ref)
            controls=[repetition_metric(y,np.roll(ref,shift)) for shift in [1,7,19,43,71,113,149]]
            # A screening rule, not a calibrated statistical significance claim.
            candidate=bool(score>.2 and score>max(controls)+.1)
            row={'array_key':key,'subcarrier_index':int(index),'baseband_frequency_hz':float((index-531.5)*234375+cfo),'repetition_15_score':score,'shifted_reference_control_scores':controls,'candidate_periodic_structure':candidate,'decoded_bits_validated':False,'semantic_fields_decoded':False,'note':'Narrowband LS estimate and exploratory reference-template repetition test; no equalized reliable payload-bit decisions or CRC validation.'}
            results.append(row);arrays[key+'_k'+str(index)]=y.astype(np.complex64)
            print(key,'k',index,'score',round(score,4),'controls max',round(max(controls),4),'candidate',candidate,flush=True)
arrays['symbol_indices']=symbol
np.savez_compressed(artifact_path('observed_nonpilot_subcarriers.npz'),**arrays)
(artifact_path('unknown_subcarrier_probe.json')).write_text(json.dumps({'seed':20261120,'training_whole_15symbol_cycles':train_cycles.tolist(),'test_whole_15symbol_cycles':test_cycles.tolist(),'two_symbol_boundary_guards':True,'results':results,'validated_message_count':0,'warning':'Screening statistics are not calibrated false-alarm probabilities; no payload interpretation is licensed by this test.'},indent=2),encoding='utf8')
print('Completed',len(results),'exploratory nonpilot tests; no validated message.')
