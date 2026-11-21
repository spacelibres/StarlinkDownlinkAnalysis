
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
import json, hashlib, struct, time
import numpy as np
from scipy import signal, stats
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import xml.etree.ElementTree as ET

ROOT = PROJECT_ROOT
OUT = experiment_output('E001', '')
p = RAW_WAV
fs, raw = wavfile.read(p, mmap=True)
n = len(raw)
print('Reading', p.name, fs, raw.shape, raw.dtype, flush=True)
db = lambda x: 10*np.log10(np.maximum(x, 1e-30))
meta = dict(file=p.name, bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),
            sample_rate=int(fs), frames=n, channels=raw.shape[1], duration_s=n/fs, dtype=str(raw.dtype),
            sample_min=raw.min(axis=0).tolist(), sample_max=raw.max(axis=0).tolist())
v = raw.astype(np.float32)/32768
meta['channel_mean']=v.mean(axis=0,dtype=np.float64).tolist()
meta['channel_rms']=np.sqrt(np.mean(v*v,axis=0,dtype=np.float64)).tolist()
meta['channel_std']=np.std(v,axis=0,dtype=np.float64).tolist()
meta['clipped_samples']=np.sum((raw == -32768)|(raw ==32767),axis=0).tolist()
meta['zero_frames']=int(np.sum(np.all(raw==0,axis=1)))
meta['identical_consecutive_frames']=int(np.sum(np.all(raw[1:]==raw[:-1],axis=1)))
meta['unique_levels_first_million']=[len(np.unique(raw[:1000000,i])) for i in range(2)]
meta['low_byte_zero_fraction']=float(np.mean((raw[:1000000].astype(np.int32)&255)==0))
sub=v[::41].astype(np.float64)
meta['channel_correlation']=float(np.corrcoef(sub.T)[0,1])
meta['channel_kurtosis_excess']=stats.kurtosis(sub).tolist()
z=v[:,0]+1j*v[:,1]
del v
meta['complex_power_dbfs']=float(db(np.mean(abs(z)**2,dtype=np.float64)))
meta['complex_peak_dbfs']=float(db(np.max(abs(z)**2)))
meta['dc_power_fraction']=float(abs(np.mean(z,dtype=np.complex128))**2/np.mean(abs(z)**2))
# Average all data with 16384-sample Hann windows; 146.48 Hz bins.
N=16384
psum=np.zeros(N); count=0; times=[]; power=[]; spec=[]
for start in range(0,n,fs//10):
    b=z[start:min(n,start+fs//10)]
    if len(b)<N: continue
    f,P=signal.welch(b,fs,nperseg=N,noverlap=N//2,return_onesided=False,detrend=False)
    psum+=P; count+=1
    times.append((start+len(b)/2)/fs)
    power.append(float(db(np.mean(abs(b)**2))))
    spec.append(np.fft.fftshift(P))
freq=np.fft.fftshift(f); psd=np.fft.fftshift(psum/count); sdb=db(psd)
S=np.array(spec); times=np.array(times); power=np.array(power)
meta['power_time_dbfs']={'min':float(power.min()),'max':float(power.max()),'std':float(power.std()),'span':float(np.ptp(power))}
peaks,props=signal.find_peaks(sdb,prominence=3,distance=8)
order=peaks[np.argsort(sdb[peaks])[::-1]][:30]
meta['spectral_peaks']=[dict(offset_hz=float(freq[k]), assumed_rf_mhz=float(1575+freq[k]/1e6), psd_dbfs_hz=float(sdb[k]),prominence_db=float(props['prominences'][np.where(peaks==k)[0][0]])) for k in order]
bands=[]
for lo in np.arange(-1200000,1200000,100000):
    mask=(freq>=lo)&(freq<lo+100000)
    bands.append(dict(lo_hz=int(lo),hi_hz=int(lo+100000),power_dbfs=float(db(psd[mask].sum()*fs/N)),median_psd_dbfs_hz=float(np.median(sdb[mask]))))
meta['bands_100khz']=bands
cum=np.cumsum(psd)/psd.sum()
meta['total_energy_99pct_interval_hz']=[float(freq[np.searchsorted(cum,q)]) for q in (.005,.995)]
meta['psd_median_dbfs_hz']=float(np.median(sdb))
meta['psd_resolution_hz']=fs/N
meta['iq_gain_imbalance_db']=float(20*np.log10(meta['channel_std'][0]/meta['channel_std'][1]))
# Correlations at physically relevant and generic lags; normalize by total power.
c=z[:fs*2]-np.mean(z[:fs*2])
lags=[1,2,3,4,5,10,100,1000,2400,4800,24000]
meta['autocorrelation_magnitude']={str(k):float(abs(np.vdot(c[:-k],c[k:]))/np.sqrt(np.vdot(c[:-k],c[:-k]).real*np.vdot(c[k:],c[k:]).real)) for k in lags}
np.savez_compressed(artifact_path('spectrum_data.npz'),offset_hz=freq,psd=psd,time_s=times,spectrogram_psd=S,power_dbfs=power)
(artifact_path('metrics.json')).write_text(json.dumps(meta,indent=2),encoding='utf8')
print(json.dumps(meta,indent=2),flush=True)

pal=json.loads(VI_ROOT / 'assets/palette-tokens.json'.read_text())[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':pal['ink'],'axes.labelcolor':pal['ink'],'axes.edgecolor':pal['line'],'xtick.color':pal['muted'],'ytick.color':pal['muted'],'figure.facecolor':pal['surface'],'axes.facecolor':pal['card'],'grid.color':pal['line']})
fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
fig.suptitle('STARLINK IF RECORDING  /  2.4 MS/s  /  15.95 s',x=.04,ha='left',fontsize=20,fontweight='bold')
ax=axs[0,0]; ax.plot(freq/1e3,sdb,color=pal['primaryOnLight'],lw=.8); ax.set(xlabel='Baseband offset (kHz), assumed I + jQ',ylabel='PSD (dBFS/Hz)',title='01  /  Average complex spectrum'); ax.axvline(0,color=pal['planetOnLight'],ls='--',lw=1,label='Nominal Starlink center (IF)'); ax.legend(fontsize=9); ax.grid(alpha=.4)
ax=axs[0,1]; cmap=LinearSegmentedColormap.from_list('orbit',[pal['deep'],pal['primary'],pal['accent']]); im=ax.pcolormesh(freq/1e3,times,db(S),shading='auto',cmap=cmap,vmin=np.percentile(db(S),5),vmax=np.percentile(db(S),99.7),rasterized=True); ax.set(xlabel='Baseband offset (kHz)',ylabel='Time (s)',title='02  /  Spectrum over time'); fig.colorbar(im,ax=ax,label='dBFS/Hz')
ax=axs[1,0]; ax.plot(times,power,color=pal['primaryOnLight']); ax.set(xlabel='Time (s)',ylabel='Complex mean power (dBFS)',title='03  /  Power in 100 ms windows'); ax.grid(alpha=.4)
ax=axs[1,1]; a=z[:1200]; ax.plot(np.arange(len(a))/fs*1e6,a.real,color=pal['primaryOnLight'],lw=.8,label='Channel 1 (assumed I)'); ax.plot(np.arange(len(a))/fs*1e6,a.imag,color=pal['planetOnLight'],lw=.6,alpha=.7,label='Channel 2 (assumed Q)'); ax.set(xlabel='Time from file start (microseconds)',ylabel='Normalized sample amplitude',title='04  /  First 500 microseconds'); ax.legend(fontsize=9); ax.grid(alpha=.4)
fig.savefig(artifact_path('overview_base.svg'))
# Embed the actual approved outlined SVG master, preserving its original geometry.
ns='{http://www.w3.org/2000/svg}'
root=ET.parse(artifact_path('overview_base.svg')); svg=root.getroot()
logo=ET.parse(str(VI_ROOT / 'assets/horizontal-primary.svg')).getroot()
logo.set('x','918'); logo.set('y','0'); logo.set('width','220'); logo.set('height','49')
for el in logo.iter():
    role=el.get('data-stroke-role'); fill=el.get('data-color-role')
    if role=='planet': el.set('stroke',pal['planetOnLight'])
    if role=='primary': el.set('stroke',pal['primaryOnLight'])
    if fill=='primary': el.set('fill',pal['primaryOnLight'])
svg.append(logo); svg.set('width','1536'); svg.set('height','960'); root.write(artifact_path('overview.svg'),encoding='utf-8',xml_declaration=True)
import resvg_py
(artifact_path('overview.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('overview.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
(artifact_path('overview_base.svg')).unlink()
print('Saved overview and metrics',flush=True)
