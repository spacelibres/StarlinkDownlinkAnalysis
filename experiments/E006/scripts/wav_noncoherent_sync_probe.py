"""Narrowband PSS/SSS power-matched test with randomized block holdouts and controls."""

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
import json,sys
import numpy as np
from scipy.io import wavfile,loadmat
from scipy import signal

OUT=experiment_output('E006', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
c=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients'];meta=json.loads((artifact_path('signal_content_extraction.json')).read_text())['wav_frame_cadence']
slope=meta['phase_drift_samples_s'];train=set(meta['training_whole_block_indices']);test=set(meta['test_whole_block_indices']);N=3200
reg=[0,0,1,1,0,1,0];bits=[]
for _ in range(127):v=reg[2]^reg[6];reg=[v]+reg[:-1];bits.append(v)
segment=np.exp(-1j*np.pi/4-.5j*np.pi*np.cumsum(2*np.r_[0,bits[::-1]]-1))
pss=np.r_[-segment,np.tile(segment,7)];pss=np.r_[-pss[-32:],pss]
sss=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel();ssst=32*np.fft.ifft(sss);ssst=np.r_[ssst[-32:],ssst]
def narrow(a):
    result=signal.resample_poly(np.r_[np.zeros(3200),a,np.zeros(3200)],1,100,window=('kaiser',8))
    result=result[20:90];result-=result.mean();result/=np.sqrt(np.sum(abs(result)**2))
    return result.astype(np.complex64)
kernels={'PSS':narrow(np.r_[pss,np.zeros(1056)]),'SSS':narrow(np.r_[np.zeros(1056),ssst]),'PSS_SSS':narrow(np.r_[pss,ssst])}
rng=np.random.default_rng(20261120)
for j in range(12):
    randomsss=sss.copy();mask=abs(sss)>0;randomsss[mask]=np.exp(.5j*np.pi*rng.integers(0,4,mask.sum()))
    a=32*np.fft.ifft(randomsss);a=np.r_[a[-32:],a]
    kernels['SSS_control_'+str(j)]=narrow(np.r_[np.zeros(1056),a])
pss_controls_only='--pss-controls-only' in sys.argv
if pss_controls_only:
    kernels={'PSS':kernels['PSS']}
    spectrum=np.fft.fft(pss[32:])
    control_count=96 if '--extended-controls' in sys.argv else 12
    for j in range(control_count):
        useful=np.fft.ifft(abs(spectrum)*np.exp(1j*rng.uniform(-np.pi,np.pi,1024)))
        control=np.r_[-useful[-32:],useful]
        kernels['PSS_control_'+str(j)]=narrow(np.r_[control,np.zeros(1056)])
acc={key:{g:np.zeros(N) for g in ['train','test','post']} for key in kernels};counts={g:np.zeros(N) for g in ['train','test','post']}
L=fs//5
for block,pos in enumerate(range(0,len(raw)-L+1,L)):
    group='train' if block in train else 'test' if block in test else 'post' if pos/fs>=13.8 else None
    if group is None:continue
    # Guard convolution boundaries; don't treat artificial block edges as framing.
    left=max(0,pos-200);right=min(len(raw),pos+L+200)
    a=raw[left:right].astype(np.float32)/32768;t=(left+np.arange(len(a)))/fs
    phase=2*np.pi*(c[0]*t**3/3+c[1]*t*t/2+c[2]*t)
    z=((a[:,0]+1j*a[:,1])*np.exp(-1j*phase)).astype(np.complex64)
    tt=(pos+np.arange(L))/fs;coord=((pos+np.arange(L))-slope*tt)%N;lo=np.floor(coord).astype(int);fraction=coord-lo
    for key,kernel in kernels.items():
        filtered=signal.fftconvolve(z,kernel.conj()[::-1],mode='same')[pos-left:pos-left+L]
        power=abs(filtered)**2;power/=power.mean()
        acc[key][group]+=np.bincount(lo,weights=power*(1-fraction),minlength=N)+np.bincount((lo+1)%N,weights=power*fraction,minlength=N)
    counts[group]+=np.bincount(lo,weights=1-fraction,minlength=N)+np.bincount((lo+1)%N,weights=fraction,minlength=N)
    if block%20==0:print('block',block,flush=True)
profiles={key:{g:a/np.maximum(counts[g],1e-12) for g,a in groups.items()} for key,groups in acc.items()}
results=[]
for key,p in profiles.items():
    smooth={g:signal.convolve(np.r_[v[-2:],v,v[:2]],np.ones(5)/5,mode='valid') for g,v in p.items()}
    # Epoch is selected on calibration blocks, independently of test/post.
    peak=int(smooth['train'].argmax())
    results.append({'replica':key,'train_selected_phase':peak,'train_peak_ratio':float(smooth['train'][peak]),'test_same_phase_ratio':float(smooth['test'][peak]),'post_same_phase_ratio':float(smooth['post'][peak])})
summary={'method':(f'PSS specificity: exact replica versus {control_count} random-phase controls preserving useful-symbol FFT magnitudes.' if pss_controls_only else 'Noncoherent matched-filter output power, exact narrowband replicas versus 12 phase-randomized SSS controls.')+' Frame phase selected only on prior training blocks. CW DC suppressed by zero-mean kernels.','random_seed':20261120,'results':results,'sync_decoded':False,'message_fields':{}}
prefix=('wav_pss_specificity_extended' if '--extended-controls' in sys.argv else 'wav_pss_specificity') if pss_controls_only else 'wav_noncoherent_sync'
(artifact_path(prefix+'_results.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
np.savez_compressed(artifact_path(prefix+'_profiles.npz'),**{key+'_'+g:p[g] for key,p in profiles.items() for g in p})
print(json.dumps(summary,indent=2),flush=True)
