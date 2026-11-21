"""Known PSS/SSS prefix test with independent per-frame phase nuisance.

Candidate frequency grids and timing are chosen only on calibration blocks.
Spectrum-preserving randomized templates receive the same timing/grid search.
This does not assign unknown payload bits.
"""

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
import argparse,json
import numpy as np
from scipy import fft
from scipy.io import wavfile,loadmat

OUT=experiment_output('E006', '')
parser=argparse.ArgumentParser();parser.add_argument('--controls',type=int,default=96);parser.add_argument('--structured-controls',type=int,default=96);parser.add_argument('--suffix',default='')
args=parser.parse_args()
fs,raw=wavfile.read(RAW_WAV,mmap=True)
meta=json.loads((artifact_path('subband_cyclostationarity.json')).read_text(encoding='utf8'))
fold=json.loads((artifact_path('recovered_cyclic_waveform.json')).read_text(encoding='utf8'))
clock=1-fold['inherited_phase_drift_samples_s']/fs
trough=next(r for r in fold['bands'] if r['band']=='full_band')['calibration_trough_phase_us']*1e-6
epoch_phase=(trough+(1/750-302*4.4e-6)/2)*fs
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
relative=np.arange(-8,41);frames=[];blocks=[];epochs=[]
for b in range(len(raw)//(fs//5)):
    first=int(np.ceil((b*.2*fs*clock-epoch_phase)/3200))
    last=int(np.floor(((b+1)*.2*fs*clock-epoch_phase)/3200))
    ids=np.arange(max(0,first),last+1)
    pos=(ids[:,None]*3200+epoch_phase+relative[None,:])/clock
    idx=np.floor(pos).astype(int)[...,None]+np.arange(-7,9)
    valid=np.all((idx>=0)&(idx<len(raw)),axis=(1,2));idx=idx[valid];pos=pos[valid];ids=ids[valid]
    delta=pos[...,None]-idx
    w=np.sinc(delta)*np.sinc(delta/8);w/=w.sum(axis=-1,keepdims=True)
    values=raw[idx].astype(np.float32)/32768
    x=np.sum((values[...,0]+1j*values[...,1])*w,axis=-1)
    t=pos/fs;phase=2*np.pi*(coeff[0]*t**3/3+coeff[1]*t*t/2+coeff[2]*t)
    x*=np.exp(-1j*phase)
    # Suppress the locally constant moving CW; retain temporal prefix shape.
    x-=x.mean(axis=1,keepdims=True)
    frames.extend(x);blocks.extend([b]*len(x));epochs.extend((ids*3200+epoch_phase)/clock/fs)
X=np.array(frames);block=np.array(blocks);epochs=np.array(epochs)
cal=np.isin(block,meta['calibration_blocks']);held=np.isin(block,meta['heldout_blocks'])
rng=np.random.default_rng(20261201);postperm=rng.permutation(meta['post_blocks']);noise=np.isin(block,postperm[:4]);post=np.isin(block,postperm[4:])
def covariance(values):return values.T@values.conj()/len(values)
Rnoise=covariance(X[noise]);e,U=np.linalg.eigh(Rnoise);floor=float(np.trace(Rnoise).real/len(relative)*.001)
W=(U/np.sqrt(np.maximum(e,floor))[None,:])@U.conj().T
Rcal=W@covariance(X[cal])@W;Rheld=W@covariance(X[held])@W;Rpost=W@covariance(X[post])@W

# Reproduce the published generator, but do not execute downloaded MATLAB code.
reg=[0,0,1,1,0,1,0];bits=[]
for _ in range(127):
    v=reg[2]^reg[6];reg=[v]+reg[:-1];bits.append(v)
segment=np.exp(-1j*np.pi/4-.5j*np.pi*np.cumsum(2*np.r_[0,bits[::-1]]-1))
useful=np.r_[-segment,np.tile(segment,7)]
pss=np.r_[-useful[-32:],useful]
sss_frequency=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel()
sss=32*fft.ifft(sss_frequency)
sss=np.r_[sss[-32:],sss]
active=np.r_[pss,sss]
taper=np.ones(1056);taper[:16]=np.arange(16)/16;taper[-16:]=np.arange(16,0,-1)/16
active*=np.r_[taper,taper]
fullrate=240000000.;N=80000;tfull=(np.arange(N)-N//2)/fullrate
frequency=fft.fftfreq(N,1/fullrate)
receiver_center=-float(np.polyval(coeff,float(epochs[cal].mean())))
distance=abs(frequency-receiver_center)
response=np.clip((1200000-distance)/150000,0,1)
times=(np.arange(N//25)-N//50)/(fullrate/25)
offsets=np.arange(-4,4.0001,.1)*1e-6
carrier_offsets=np.array([-117187.5,0,117187.5])
sample_times=relative/fs

def candidates(wave):
    vectors=[];labels=[]
    padded=np.zeros(N,complex);padded[N//2:N//2+len(wave)]=wave
    for shift in carrier_offsets:
        narrowed=fft.ifft(fft.fft(padded*np.exp(2j*np.pi*shift*tfull))*response)[::25]
        for offset in offsets:
            tx=sample_times-offset
            v=np.interp(tx,times,narrowed.real)+1j*np.interp(tx,times,narrowed.imag)
            v-=v.mean();v=W@v;v/=np.linalg.norm(v)
            vectors.append(v);labels.append((float(shift),float(offset)))
    return np.array(vectors),labels

def best(wave):
    v,labels=candidates(wave)
    score=np.einsum('ni,ij,nj->n',v.conj(),Rcal,v,optimize=True).real
    k=int(np.argmax(score));selected=v[k]
    heldout=float(np.vdot(selected,Rheld@selected).real)
    after=float(np.vdot(selected,Rpost@selected).real)
    return {'carrier_offset_from_moving_line_hz':labels[k][0],'offset_from_power_guard_epoch_s':labels[k][1],
            'calibration_whitened_energy':float(score[k]),'heldout_whitened_energy':heldout,
            'post_whitened_energy':after},selected

exact,v=best(active)
controls=[];spectrum=fft.fft(active)
for j in range(args.controls):
    control=fft.ifft(abs(spectrum)*np.exp(1j*rng.uniform(-np.pi,np.pi,len(active))))
    result,_=best(control);controls.append(result)
    if j%24==0:print('template control',j,flush=True)
tail=sum(c['heldout_whitened_energy']>=exact['heldout_whitened_energy'] for c in controls)
structured=[]
for j in range(args.structured_controls):
    randomized_segment=fft.ifft(abs(fft.fft(segment))*np.exp(1j*rng.uniform(-np.pi,np.pi,128)))
    useful_control=np.r_[-randomized_segment,np.tile(randomized_segment,7)]
    pss_control=np.r_[-useful_control[-32:],useful_control]
    sss_control_frequency=np.zeros(1024,complex);loaded=abs(sss_frequency)>0
    sss_control_frequency[loaded]=np.exp(.5j*np.pi*rng.integers(0,4,loaded.sum()))
    sss_control=32*fft.ifft(sss_control_frequency);sss_control=np.r_[sss_control[-32:],sss_control]
    wave=np.r_[pss_control,sss_control]*np.r_[taper,taper]
    result,_=best(wave);structured.append(result)
    if j%48==0:print('structure-preserving control',j,flush=True)
Rsignal=(Rcal-np.eye(len(relative)))
eigen=np.linalg.eigvalsh(Rsignal)[::-1]
report={'scope':'Existing WAV narrowband PSS+SSS hypothesis; no decoded header or payload.',
        'epoch_from_power_guard_phase_us':epoch_phase/fs*1e6,'frame_window_samples':relative.tolist(),
        'candidate_frequency_offsets_hz':carrier_offsets.tolist(),'timing_offsets_us':[-4,4],
        'per_frame_constant_removed':True,'calibration_blocks':meta['calibration_blocks'],'heldout_blocks':meta['heldout_blocks'],
        'post_noise_calibration_blocks':postperm[:4].tolist(),'post_heldout_blocks':postperm[4:].tolist(),
        'seed':20261201,'exact':exact,'randomized_controls':controls,'controls_exceeding_exact':tail,
        'control_count':len(controls),'random_template_rank_fraction':(tail+1)/(len(controls)+1),
        'structure_preserving_controls':structured,
        'structured_controls_exceeding_exact':sum(c['heldout_whitened_energy']>=exact['heldout_whitened_energy'] for c in structured),
        'structured_control_count':len(structured),
        'control_heldout_energy_max':max(c['heldout_whitened_energy'] for c in controls),
        'calibration_signal_covariance_top_eigenvalues':eigen[:8].tolist(),
        'known_sequence_confirmed':False,
        'limitations':['Narrowband templates depend on the RF center versus moving-line convention and unknown receiver response.','Earlier clock and chirp models are inherited; this is not end-to-end blind synchronization.','Random controls preserve whole-prefix FFT magnitudes but do not exhaust all alternative waveforms.','Whitening uses separate post-event blocks. Per-frame phase is not forced to match the CW.','A high energy at the cycle boundary is not sufficient to identify PSS or SSS.']}
prefix='narrowband_sync_prefix'+('_'+args.suffix if args.suffix else '')
(artifact_path(prefix+'_test.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
projection=(X@W.T)@v.conj()
np.savez_compressed(artifact_path(prefix+'_observations.npz'),prefix_iq=X.astype(np.complex64),frame_epoch_s=epochs,block_index=block,
                    reference_relative_sample=relative,selected_whitened_template=v.astype(np.complex64),selected_projection=projection.astype(np.complex64))
print(json.dumps({'exact':exact,'controls_exceeding_exact':tail,'control_count':len(controls),
                  'control_heldout_energy_max':report['control_heldout_energy_max'],'top_signal_eigenvalues':eigen[:8].tolist()},indent=2),flush=True)
print('Structured controls exceeding exact:',report['structured_controls_exceeding_exact'],'/',len(structured),flush=True)
