
# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E001')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json, math
import numpy as np
from scipy.io import wavfile
from scipy import signal, fft, stats

out=experiment_output('E001', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
R={}
for ch in range(2):
    u,c=np.unique(raw[:1000000,ch],return_counts=True)
    R[f'channel_{ch+1}_quantization']={'unique_levels':len(u),'levels':u.tolist(),'most_common_step':int(stats.mode(np.diff(u),keepdims=False).mode)}
# Narrowband precision measurement on 0.25 s snapshots distributed through file.
rows=[]
for t in [0,2,4,6,8,10,12,14,15.5]:
    a=raw[int(t*fs):int((t+.25)*fs)].astype(np.float32)/32768
    z=a[:,0]+1j*a[:,1]
    f,p=signal.welch(z,fs,nperseg=262144,noverlap=131072,return_onesided=False,detrend=False)
    mask=(f>10000)&(f<16000); inds=np.where(mask)[0]; k=inds[np.argmax(p[mask])]
    floor=float(np.median(p[(f>20000)&(f<50000)]))
    m=(f>f[k]-1500)&(f<f[k]+1500)
    excess=np.maximum(p[m]-floor,0)
    rows.append({'time_s':t,'peak_offset_hz':float(f[k]),'peak_above_local_floor_db':float(10*np.log10(p[k]/floor)),'excess_power_dbfs_3khz':float(10*np.log10(np.sum(excess)*fs/262144))})
R['tone_snapshots']=rows

# Independently implement the standard G1/G2 LFSRs using PRN output taps.
taps=[(2,6),(3,7),(4,8),(5,9),(1,9),(2,10),(1,8),(2,9),(3,10),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),(9,10),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(1,3),(4,6),(5,7),(6,8),(7,9),(8,10),(1,6),(2,7),(3,8),(4,9)]
codes=[]
for ta,tb in taps:
    a=np.ones(10,dtype=int); b=a.copy(); code=[]
    for _ in range(1023):
        code.append(1-2*(a[9]^b[ta-1]^b[tb-1]))
        fa=a[2]^a[9]; fb=b[1]^b[2]^b[5]^b[7]^b[8]^b[9]
        a[1:]=a[:-1]; a[0]=fa; b[1:]=b[:-1]; b[0]=fb
    codes.append(np.array(code,dtype=np.float32))
N=fs//1000; K=20
idx=np.floor(np.arange(N)*1023000/fs).astype(int)
C=fft.fft(np.array([c[idx] for c in codes]),axis=-1).conj()
# Generator sanity: length, periodic autocorrelation and first ten PRN1 bits (octal 1440).
assert len(codes[0])==1023
assert ''.join('1' if x<0 else '0' for x in codes[0][:10])=='1100100000'
ac=np.fft.ifft(abs(np.fft.fft(codes[0]))**2).real
assert abs(ac[0]-1023)<.01 and np.max(abs(ac[1:]))<66
freqs=np.arange(-15000,15001,500)
t=np.arange(N)/fs
def search(z,center):
    best=np.zeros(32); bestfreq=np.zeros(32); bestphase=np.zeros(32,dtype=int)
    for dop in freqs:
        mix=np.exp(-2j*np.pi*(center+dop)*t).astype(np.complex64)
        X=fft.fft(z.reshape(K,N)*mix,axis=-1)
        # Per-millisecond constant carrier phase cancels in noncoherent power.
        corr=fft.ifft(C[:,None,:]*X[None,:,:],axis=-1)
        P=np.sum(abs(corr)**2,axis=1)
        ratios=P/P.mean(axis=1,keepdims=True)
        phases=ratios.argmax(axis=1); vals=ratios[np.arange(32),phases]
        hit=vals>best
        best[hit]=vals[hit]; bestfreq[hit]=dop; bestphase[hit]=phases[hit]
    return [{'prn':int(i+1),'peak_to_phase_mean':float(best[i]),'frequency_residual_hz':float(bestfreq[i]),'code_phase_sample':int(bestphase[i])} for i in np.argsort(best)[::-1]]
# Synthetic signal positive control, same search and real recording noise.
a=raw[fs:fs+K*N].astype(np.float32)/32768
z=a[:,0]+1j*a[:,1]
tt=np.arange(K*N)/fs
fake=np.tile(np.roll(codes[6][idx],375),K)*np.exp(2j*np.pi*(420000+2500)*tt)*.003
control=search((z+fake).astype(np.complex64),420000)
assert control[0]['prn']==7 and abs(control[0]['frequency_residual_hz']-2500)<=500
R['acquisition_positive_control']=control[0]
R['gps_search']={'prns':'1-32','center_offsets_hz':[420000,-420000],'residual_range_hz':[-15000,15000],'frequency_step_hz':500,'coherent_ms':1,'noncoherent_sums':K,'segments_s':[1,8], 'note':'Exploratory peak/mean statistics, not a calibrated GNSS receiver or C/N0 estimate.'}
R['acquisition_results']=[]
for sec in [1,8]:
    a=raw[sec*fs:sec*fs+K*N].astype(np.float32)/32768; z=a[:,0]+1j*a[:,1]
    for center in [420000,-420000]:
        result=search(z,center)
        row={'time_s':sec,'center_offset_hz':center,'top_results':result[:10]}
        R['acquisition_results'].append(row)
        print(json.dumps(row),flush=True)
R['ideal_white_noise_familywise_1pct_threshold']=float(stats.gamma.isf(.01/(32*len(freqs)*N*4),a=K,scale=1/K))
(artifact_path('deeper_metrics.json')).write_text(json.dumps(R,indent=2),encoding='utf8')
print('Done',flush=True)
