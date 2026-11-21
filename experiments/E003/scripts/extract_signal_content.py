"""Recover observable symbol structure; never label model IDs or random bytes as payload."""

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
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

OUT=experiment_output('E003', '')
SEED=20261120
archive=np.load(artifact_path('decoded_known_pilot_symbols.npz'),allow_pickle=False)
states=archive['expected_states'];expected=np.exp(.5j*np.pi*(states+.5))
meta=json.loads((artifact_path('known_pilot_verification.json')).read_text())
selection=json.loads(LEO_SNAPSHOT / 'reports/2026_10_18_dsx01_pilot_clock/DSX01-S05/selection.json'.read_text())
tones=(np.arange(8)-3.5)*234375
times=(np.arange(300)+2.5)*4.4e-6;times-=times.mean()
rng=np.random.default_rng(SEED)
# Whole 30-symbol blocks, with boundary guards, form complementary calibration folds.
order=rng.permutation(10);calibration=np.isin(np.arange(300)//30,order[:5])
interior=(np.arange(300)%30>=2)&(np.arange(300)%30<28)
def crossfit(cubes):
    numerator=np.zeros((300,8),complex);denominator=np.zeros((300,8))
    for train in [calibration,~calibration]:
        train=train&interior;test=(~train)&interior
        # Exclude the boundary guard symbols from both sets.
        test=(~np.isin(np.arange(300)//30,np.unique(np.arange(300)[train]//30)))&interior
        for key,cube,cfo in cubes:
            grid=np.arange(-1000,1001,20)
            phasors=np.exp(-2j*np.pi*grid[:,None]*times)
            matched=cube[:,train,:]*expected[None,train,:].conj()
            cc=np.einsum('ds,fst->dft',phasors[:,train],matched,optimize=True)
            k=int(np.sum(abs(cc)**2,axis=(1,2)).argmax())
            corrected=cube*phasors[k][None,:,None]
            h=np.mean(corrected[:,train,:]*expected[None,train,:].conj(),axis=1)
            residual=corrected[:,train,:]-h[:,None,:]*expected[None,train,:]
            noise=np.mean(abs(residual)**2,axis=1)
            significant=abs(h)**2>9*noise/train.sum()
            significant &= (abs(cfo+tones)<.95*1250000)[None,:]
            weight=significant/np.maximum(noise,1e-12)
            numerator[test]+=np.einsum('ft,fst->st',h.conj()*weight,corrected[:,test,:])
            denominator[test]+=np.sum(abs(h)**2*weight,axis=0)[None,:]
    eq=numerator/np.maximum(denominator,1e-12)
    hard=np.rint((np.angle(eq)-np.pi/4)/(np.pi/2)).astype(int)%4
    supported=interior[:,None]&(denominator>0)
    return eq,hard,supported

cubes=[]
for w in meta['windows']:
    key=w['array_key'];cfo=selection['windows'][int(key[3:])]['winner']['fractional_tracking_cfo_hz']
    cubes.append((key,archive[key+'_pilot_cube'].astype(complex),cfo))
eq,hard,supported=crossfit(cubes)
noise_rng=np.random.default_rng(SEED+1)
null=[(k,(noise_rng.normal(size=v.shape)+1j*noise_rng.normal(size=v.shape))/np.sqrt(2),cfo) for k,v,cfo in cubes]
null_eq,null_hard,null_supported=crossfit(null)
accuracy=float(np.mean(hard[supported]==np.broadcast_to(states,hard.shape)[supported]))
null_accuracy=float(np.mean(null_hard[null_supported]==np.broadcast_to(states,hard.shape)[null_supported])) if np.any(null_supported) else None
with (artifact_path('observed_pilot_states.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['ofdm_symbol_index','subcarrier_index','observed_state_0_3','published_reference_state','supported','equalized_real','equalized_imag','meaning'])
    for s in range(300):
        for tone in range(8):writer.writerow([s+2,tone+528,int(hard[s,tone]) if supported[s,tone] else '',int(states[s,tone]),bool(supported[s,tone]),eq[s,tone].real,eq[s,tone].imag,'universal_known_pilot_not_message_field'])
sequences={str(k+528):''.join(str(hard[s,k]) if supported[s,k] else '?' for s in range(300)) for k in range(8)}
np.savez_compressed(artifact_path('observed_pilot_states.npz'),equalized=eq,observed_states=hard,reference_states=states,supported=supported,calibration_blocks=order[:5])

# Narrowband WAV: test the expected 750-Hz frame cadence through a power-gap fold.
fs,raw=wavfile.read(RAW_WAV,mmap=True)
period=3200;blockframes=150;length=period*blockframes
profiles=[];block_times=[]
for pos in range(0,len(raw)-length+1,length):
    a=raw[pos:pos+length].astype(np.float64)
    p=np.sum(a*a,axis=1).reshape(blockframes,period)
    p/=p.mean(axis=1,keepdims=True)
    profiles.append(p.mean(axis=0));block_times.append((pos+length/2)/fs)
profiles=uniform_filter1d(np.array(profiles),size=11,axis=1,mode='wrap');block_times=np.array(block_times)
eligible=np.where(block_times<13)[0];perm=np.random.default_rng(SEED).permutation(eligible)
train=perm[:len(perm)//2];test=perm[len(perm)//2:];post=np.where(block_times>13.8)[0]
x=np.arange(period);best=None
for slope in np.arange(-40,40.01,.5):
    shifted=np.array([np.interp((x+slope*t)%period,x,profiles[j],period=period) for j,t in enumerate(block_times)])
    mean=shifted[train].mean(axis=0);phase=int(mean.argmin());depth=1-mean[phase]
    if best is None or depth>best[0]:best=(depth,slope,phase,shifted)
depth,slope,phase,shifted=best;test_profile=shifted[test].mean(axis=0);post_profile=shifted[post].mean(axis=0)
fold={'assumed_nominal_frame_hz':750,'fitted_frame_hz':float((fs-slope)/period),'fitted_frame_period_us':float(period/(fs-slope)*1e6),'phase_drift_samples_s':float(slope),'folded_gap_phase_sample':phase,'training_gap_depth_fraction':float(depth),'test_gap_depth_fraction':float(1-test_profile[phase]),'post_event_gap_depth_fraction':float(1-post_profile[phase]),'random_seed':SEED,'training_whole_block_indices':train.tolist(),'test_whole_block_indices':test.tolist(),'block_duration_s':length/fs,'smoothing_samples':11,'scope':'Expected-cadence search, not a blind universal protocol detector; fitted rate mixes receiver and transmitter clocks. No frame-header bits decoded.'}
np.savez_compressed(artifact_path('wav_frame_structure.npz'),time_s=block_times,profiles=profiles,shifted_profiles=shifted,phase_sample=phase,slope_samples_s=slope,train_indices=train,test_indices=test,post_indices=post)
result={'decoded_message_count':0,'semantic_fields':{},'pilot':{'kind':'reference-assisted measured universal pilot states','state_serialization':'0..3 phase states; not a specified payload bit mapping','observed_states':int(supported.sum()),'known_reference_agreement':accuracy,'noise_control_supported_states':int(null_supported.sum()),'noise_control_agreement':null_accuracy,'sequences':sequences,'guard_symbols_are_erasures':True,'warning':'Repeated known pilots were combined across source windows. This does not recover varying payload or messages. All phase/frequency fits use calibration blocks only.'},'wav_frame_cadence':fold}
(artifact_path('signal_content_extraction.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print('Observed known states:',supported.sum(),'agreement',accuracy,'noise supported:',null_supported.sum(),'noise agreement',null_accuracy,flush=True)
print('First 80 measured states, k528:',sequences['528'][:80],flush=True)
print('WAV cadence',json.dumps(fold),flush=True)
assert accuracy>.9 and supported.sum()>1500
assert fold['test_gap_depth_fraction']>.04
print('PASS: measured pilot recovery and randomized-block frame-fold check; decoded semantic messages = 0.',flush=True)
