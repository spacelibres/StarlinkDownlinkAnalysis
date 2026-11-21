"""Recover periodic power structure before attempting unknown symbol decisions."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E005')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv,json
import numpy as np
from scipy import signal,fft,ndimage
from scipy.io import wavfile

OUT=experiment_output('E005', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
meta=json.loads((artifact_path('subband_cyclostationarity.json')).read_text(encoding='utf8'))
cal=np.array(meta['calibration_blocks']);test=np.array(meta['heldout_blocks']);post=np.array(meta['post_blocks'])
clock=json.loads((artifact_path('signal_content_extraction.json')).read_text(encoding='utf8'))['wav_frame_cadence']
slope=clock['phase_drift_samples_s'];N=3200;L=fs//5;count=len(raw)//L
bands={'negative_lobe':(-950000,-650000),'positive_lobe':(500000,750000),'center_control':(-100000,100000)}
profiles={k:[] for k in ['full_band',*bands]}
filters={k:signal.firwin(513,(hi-lo)/2,fs=fs,window=('kaiser',8)).astype(np.float32) for k,(lo,hi) in bands.items()}
for b in range(count):
    pos=b*L;left=max(0,pos-4800);right=min(len(raw),pos+L+4800)
    a=raw[left:right].astype(np.float32)/32768;t=(left+np.arange(len(a)))/fs
    x=a[:,0]+1j*a[:,1];tt=(pos+np.arange(L))/fs
    coordinate=((pos+np.arange(L))-slope*tt)%N
    lower=np.floor(coordinate).astype(int);frac=coordinate-lower
    weights=np.bincount(lower,weights=1-frac,minlength=N)+np.bincount((lower+1)%N,weights=frac,minlength=N)
    def add(name,power):
        power=power/power.mean()
        folded=np.bincount(lower,weights=power*(1-frac),minlength=N)+np.bincount((lower+1)%N,weights=power*frac,minlength=N)
        profiles[name].append(folded/weights)
    add('full_band',abs(x[pos-left:pos-left+L])**2)
    for name,(lo,hi) in bands.items():
        shifted=(x*np.exp(-2j*np.pi*(lo+hi)/2*t)).astype(np.complex64)
        filtered=signal.fftconvolve(shifted,filters[name],mode='same')[pos-left:pos-left+L]
        add(name,abs(filtered)**2)
    if b%20==0:print('fold block',b,flush=True)
arrays={'phase_us':np.arange(N)/fs*1e6};rows=[];correlations=[]
for name,pp in profiles.items():
    p=np.array(pp);smooth=ndimage.uniform_filter1d(p,11,axis=1,mode='wrap')
    c=smooth[cal].mean(axis=0);v=smooth[test].mean(axis=0);after=smooth[post].mean(axis=0)
    fitted=c-1;fitted/=np.linalg.norm(fitted)
    cross=fft.irfft(fft.rfft(smooth[test]-1,axis=1)*fft.rfft(fitted).conj()[None,:],n=N,axis=1)
    correlations.append(cross)
    trough=int(np.argmin(c));peak=int(np.argmax(c))
    row={'band':name,'calibration_trough_phase_us':trough/fs*1e6,'calibration_trough_power_ratio':float(c[trough]),
         'heldout_at_calibration_trough_power_ratio':float(v[trough]),'post_at_same_trough_power_ratio':float(after[trough]),
         'calibration_peak_phase_us':peak/fs*1e6,'heldout_at_calibration_peak_power_ratio':float(v[peak]),
         'train_test_profile_correlation':float(np.corrcoef(c-1,v-1)[0,1]),
         'train_post_profile_correlation':float(np.corrcoef(c-1,after-1)[0,1]),
         'heldout_matched_profile_score':float(cross[:,0].mean())}
    rows.append(row);arrays[name+'_block_profiles']=p.astype(np.float32)
    for key,value in [('calibration_profile',c),('heldout_profile',v),('post_profile',after),('heldout_standard_error',smooth[test].std(axis=0,ddof=1)/np.sqrt(len(test)))]:arrays[name+'_'+key]=value
rng=np.random.default_rng(20261130);shifts=rng.integers(0,N,size=(999,len(test)))
null=[]
for cross in correlations:null.append(cross[np.arange(len(test))[None,:],shifts].mean(axis=1))
null=np.array(null).T;scale=null.std(axis=0,ddof=1)
joint=(null/scale).max(axis=1)
for j,row in enumerate(rows):
    stat=row['heldout_matched_profile_score']/scale[j]
    row['heldout_random_shift_standardized_score']=float(stat)
    row['four_band_family_random_shift_p']=float((1+np.sum(joint>=stat))/1000)
arrays['control_whole_block_shifts']=shifts;arrays['joint_max_shift_control']=joint
np.savez_compressed(artifact_path('recovered_cyclic_waveform.npz'),**arrays)
result={'scope':'Existing original WAV only; periodic power structure, not message bits.',
        'period_samples':N,'nominal_period_us':N/fs*1e6,'inherited_phase_drift_samples_s':slope,
        'known_clock_model_not_new_blind_acquisition':True,'smoothing_samples':11,'seed':20261130,
        'calibration_blocks':cal.tolist(),'heldout_blocks':test.tolist(),'post_blocks':post.tolist(),'bands':rows,
        'limitations':['Power profiles do not determine complex waveform or modulation.','The clock model was fitted in earlier work; this test measures repeatability under that model.','Zero-phase filters retain edge spread and unknown front-end delay.','A trough is a candidate guard/low-power interval until known sequence alignment establishes frame origin.']}
(artifact_path('recovered_cyclic_waveform.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
with (artifact_path('recovered_cyclic_power.csv')).open('w',encoding='utf-8-sig',newline='') as file:
    fields=['phase_us']+[name+'_'+group for name in profiles for group in ['calibration','heldout','post']]
    writer=csv.writer(file);writer.writerow(fields)
    for j,phase in enumerate(arrays['phase_us']):writer.writerow([phase]+[arrays[name+'_'+group+'_profile'][j] for name in profiles for group in ['calibration','heldout','post']])
print(json.dumps(result['bands'],indent=2),flush=True)
