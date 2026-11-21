"""Whole-record component relationships and blind envelope-rate survey.

Offline only. No identifier, orbit, header interpretation or 750-Hz target is
used to select envelope spectral peaks. Independent whole blocks are held out.
"""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E004')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv,json
import numpy as np
from scipy import fft,signal,ndimage
from scipy.io import wavfile

OUT=experiment_output('E004', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
event=13.635
bands={'negative_lobe':(-950000,-650000),'positive_lobe':(500000,750000),
       'center':(-100000,100000),'fixed_13k':(8000,18000),
       'quiet_negative':(-400000,-300000),'upper_edge':(1050000,1150000)}

# Non-overlapping FFTs retain absolute time. These bands are descriptive regions,
# not claims that each region contains a single physical transmitter.
N=2048;f=fft.fftfreq(N,1/fs);window=signal.windows.hann(N,sym=False).astype(np.float32)
weights={k:((f>=lo)&(f<=hi)) for k,(lo,hi) in bands.items()}
step=fs//50;times=[];powers=[];envelopes=[]
for pos in range(0,len(raw)-step+1,step):
    a=raw[pos:pos+step].astype(np.float32)/32768
    z=a[:,0]+1j*a[:,1]
    envelopes.append((abs(z)**2).reshape(-1,200).mean(axis=1))
    usable=len(z)//N*N
    p=(abs(fft.fft(z[:usable].reshape(-1,N)*window,axis=1))**2).mean(axis=0)
    p/=N*np.sum(window**2)
    powers.append([p[weights[k]].sum() for k in bands]);times.append((pos+step/2)/fs)
times=np.array(times);powers=np.array(powers);envelope=np.concatenate(envelopes)
pre=(times>=.5)&(times<=12.5);post=(times>=13.9)&(times<=15.5)
rows=[]
for col,key in enumerate(bands):
    x=powers[:,col];choices=[]
    for t in np.arange(12.8,14.31,.02):
        before=x[(times>=t-.4)&(times<t-.04)].mean()
        after=x[(times>=t+.04)&(times<t+.4)].mean()
        choices.append((float(10*np.log10(after/before)),float(t)))
    largest=min(choices)
    rows.append({'component':key,'band_hz':list(bands[key]),
                 'whole_pre_post_change_db':float(10*np.log10(x[post].mean()/x[pre].mean())),
                 'local_event_change_db':float(10*np.log10(x[(times>=13.8)&(times<14.1)].mean()/x[(times>=13.1)&(times<13.5)].mean())),
                 'strongest_local_drop_center_s':largest[1],
                 'strongest_local_drop_db':largest[0]})

# Descriptive correlations use levels and first differences separately, and the
# common termination is excluded. Correlation is not proof of a shared source.
prelog=np.log(powers[pre]);levelcorr=np.corrcoef(prelog.T)
diffcorr=np.corrcoef(np.diff(prelog,axis=0).T)

# Blind search over 50..5500 Hz, with 5-Hz bins, on random whole 0.2-s blocks.
ef=fs/200;L=int(ef/5);blockcount=len(envelope)//L
e=envelope[:blockcount*L].reshape(blockcount,L)
e=e/e.mean(axis=1,keepdims=True)-1
w=signal.windows.hann(L,sym=False)
spectra=abs(fft.rfft(e*w,axis=1))**2/np.sum(w*w)
freq=fft.rfftfreq(L,1/ef)
seed=20261120;perm=np.random.default_rng(seed).permutation(np.arange(65))
train=perm[:32];test=perm[32:];postids=np.flatnonzero((np.arange(blockcount)+.5)/5>=13.9)
profiles={'train':spectra[train].mean(axis=0),'test':spectra[test].mean(axis=0),'post':spectra[postids].mean(axis=0)}
baseline={k:ndimage.median_filter(v,size=31,mode='nearest') for k,v in profiles.items()}
contrast={k:v/np.maximum(baseline[k],1e-30) for k,v in profiles.items()}
peaks,_=signal.find_peaks(contrast['train'],distance=3)
peaks=[i for i in peaks if 50<=freq[i]<=5500]
peaks=sorted(peaks,key=lambda i:-contrast['train'][i])[:12]
rates=[{'frequency_hz':float(freq[i]),'train_local_power_ratio':float(contrast['train'][i]),
        'heldout_local_power_ratio':float(contrast['test'][i]),'post_local_power_ratio':float(contrast['post'][i])} for i in peaks]

# Spectral change is estimated in original receiver coordinates. Negative
# differences remain negative; this is not a fitted signal/noise separation.
with np.load(artifact_path('waterfall_detail_data.npz'),allow_pickle=False) as z:
    sf=z['frequency_hz'];st=z['time_s'];sp=z['psd']
    before=sp[(st>=.5)&(st<=12.5)].mean(axis=0);after=sp[(st>=13.9)&(st<=15.5)].mean(axis=0)
    excess=ndimage.gaussian_filter1d(before-after,100)
    axes=[]
    for center in np.arange(-300000,300001,1000):
        distances=np.arange(20000,850001,2000)
        left=np.interp(center-distances,sf,excess);right=np.interp(center+distances,sf,excess)
        axes.append(float(np.corrcoef(left,right)[0,1]))
    axes=np.array(axes);axisgrid=np.arange(-300000,300001,1000)
    best=int(np.nanargmax(axes))
    symmetry={'best_reflection_axis_hz':int(axisgrid[best]),'shape_correlation':float(axes[best]),
              'note':'Descriptive scan of smoothed pre-minus-post spectrum; axis selected on the same data. Not a validated RF channel center, image diagnosis, or Doppler estimate.'}
np.savez_compressed(artifact_path('global_signal_audit_arrays.npz'),time_s=times,band_power=powers,
    envelope_frequency_hz=freq,**{k+'_envelope_spectrum':v for k,v in profiles.items()},
    receiver_frequency_hz=sf,pre_spectrum=before,post_spectrum=after,smoothed_difference=excess,
    reflection_axis_hz=axisgrid,reflection_correlation=axes)
result={'scope':'Existing local WAV only; receive equipment and actual RF/LNB configuration remain unverified. No acquisition or transmission.',
        'sample_rate_hz':int(fs),'event_time_s':event,'components':rows,
        'pre_event_level_correlation':levelcorr.tolist(),'pre_event_first_difference_correlation':diffcorr.tolist(),
        'correlation_component_order':list(bands),'blind_envelope_rates':rates,
        'validation':{'seed':seed,'training_whole_200ms_blocks':train.tolist(),'heldout_whole_200ms_blocks':test.tolist(),'post_event_blocks':postids.tolist(),'candidate_selection':'Only training spectral peaks; no nominal frame frequency is used.'},
        'spectral_symmetry_scan':symmetry,
        'limitations':['Component associations are observational, not transmitter identities.','The post-event interval can contain other transmissions and receiver noise.','Power-envelope peaks identify repetition, not bit or message semantics.','The original WAV and pilot snippets are different captures and cannot be combined as one waveform.']}
(artifact_path('global_signal_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
with (artifact_path('global_component_power.csv')).open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.writer(file);writer.writerow(['time_s']+list(bands));writer.writerows(zip(times,*powers.T))
print(json.dumps({'components':rows,'blind_rates':rates,'symmetry':symmetry,'first_difference_correlation':dict(zip(list(bands)[1:],diffcorr[0,1:]))},indent=2),flush=True)
