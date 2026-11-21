"""Carrier-aided coherent frame folding of the original narrowband WAV."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E006')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json
import numpy as np
from scipy.io import wavfile,loadmat
from scipy import signal,fft

OUT=experiment_output('E006', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
base=json.loads((artifact_path('signal_content_extraction.json')).read_text())['wav_frame_cadence']
slope=base['phase_drift_samples_s'];N=3200
train=set(base['training_whole_block_indices']);test=set(base['test_whole_block_indices'])
def modelphase(t):return 2*np.pi*(coeff[0]*t**3/3+coeff[1]*t*t/2+coeff[2]*t)
low=[]
for pos in range(0,len(raw),fs):
    a=raw[pos:min(pos+fs,len(raw))].astype(np.float32)/32768
    t=(pos+np.arange(len(a)))/fs
    z=(a[:,0]+1j*a[:,1])*np.exp(-1j*modelphase(t))
    low.append(signal.resample_poly(z,1,120))
low=np.concatenate(low)
smooth=signal.sosfiltfilt(signal.butter(4,100,fs=20000,output='sos'),low)
phase=np.unwrap(np.angle(smooth));lowtime=np.arange(len(low))/20000
groups={name:{'sum':np.zeros(N,complex),'weight':np.zeros(N)} for name in ['train','test','post','random_phase_control']}
rng=np.random.default_rng(20261120)
frame_phases=rng.uniform(-np.pi,np.pi,13000)
blocklen=fs//5
for block,pos in enumerate(range(0,len(raw)-blocklen+1,blocklen)):
    name='train' if block in train else 'test' if block in test else 'post' if pos/fs>=13.8 else None
    if name is None:continue
    a=raw[pos:pos+blocklen].astype(np.float32)/32768;t=(pos+np.arange(len(a)))/fs
    z=(a[:,0]+1j*a[:,1])*np.exp(-1j*(modelphase(t)+np.interp(t,lowtime,phase)))
    # Retain only frame-varying structure; a stationary carrier is not content.
    z-=z.mean()
    coordinate=((pos+np.arange(len(a)))-slope*t)%N
    left=np.floor(coordinate).astype(int);frac=coordinate-left
    def add(which,values):
        for idx,weight in [(left,1-frac),((left+1)%N,frac)]:
            groups[which]['sum']+=np.bincount(idx,weights=(values*weight).real,minlength=N)+1j*np.bincount(idx,weights=(values*weight).imag,minlength=N)
            groups[which]['weight']+=np.bincount(idx,weights=weight,minlength=N)
    add(name,z)
    if name=='test':
        frameid=np.floor(((pos+np.arange(len(a)))-slope*t)/N).astype(int)
        add('random_phase_control',z*np.exp(1j*frame_phases[frameid]))
folds={k:v['sum']/np.maximum(v['weight'],1e-12) for k,v in groups.items()}
def coherence(a,b):return float(abs(np.vdot(a,b))/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real))
report={'method':'Carrier phase tracked at 20 kS/s with 100-Hz zero-phase lowpass; frame folding uses prior randomized whole-block assignment. DC carrier removed per 0.2-s block.','full_frame_train_test_coherence':coherence(folds['train'],folds['test']),'random_frame_phase_control_coherence':coherence(folds['train'],folds['random_phase_control']),'post_event_coherence':coherence(folds['train'],folds['post']),'frame_period_samples':N,'observed_sync_sequence_decoded':False,'message_fields':{}}

# Independent known PSS+SSS replica, filtered to the recorded 2.4-MHz bandwidth.
reg=[0,0,1,1,0,1,0];bits=[]
for _ in range(127):
    val=reg[2]^reg[6];reg=[val]+reg[:-1];bits.append(val)
seq=2*np.r_[0,bits[::-1]]-1
segment=np.exp(-1j*np.pi/4-.5j*np.pi*np.cumsum(seq))
pss=np.r_[-segment,np.tile(segment,7)];pss=np.r_[-pss[-32:],pss]
sss=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel()
ssst=32*fft.ifft(sss);ssst=np.r_[ssst[-32:],ssst]
wave=np.zeros(320000,complex);wave[:1056]=pss;wave[1056:2112]=ssst
template=signal.resample_poly(wave,1,100,window=('kaiser',8))
template-=template.mean()
corr=fft.ifft(fft.fft(folds['train'])*fft.fft(template).conj())
lag=int(abs(corr).argmax());replica=np.roll(template,lag)
report['known_sync_replica_best_lag_sample']=lag
report['known_sync_replica_train_coherence']=coherence(folds['train'],replica)
report['known_sync_replica_test_coherence']=coherence(folds['test'],replica)
report['guard_based_frame_start_prediction_sample']=base['folded_gap_phase_sample']+fs*(4.4e-6+2/15*1e-6)/2
report['caution']='Carrier-aided folding may fail when CW and OFDM frame phases are unrelated. A periodic power gap alone is not PSS/SSS decoding. The replica assumes the moving line is the OFDM carrier center and an ideal narrowband filter.'
np.savez_compressed(artifact_path('wav_coherent_frame_probe.npz'),**folds,sync_template=template,best_replica=replica,carrier_time_s=lowtime,carrier_phase=phase)
(artifact_path('wav_coherent_frame_results.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
