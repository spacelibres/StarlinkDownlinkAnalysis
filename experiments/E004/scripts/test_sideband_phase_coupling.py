"""Distinguish power-spectrum symmetry from coherent conjugate sidebands.

Only existing WAV data are read. Receiver-zero, moving-line and previously
estimated displaced axes are hypotheses, not inferred carrier identities.
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
import json
import numpy as np
from scipy import fft,signal
from scipy.io import wavfile

OUT=experiment_output('E004', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
with np.load(artifact_path('wav_coherent_frame_probe.npz'),allow_pickle=False) as z:
    pt=z['carrier_time_s'];pp=z['carrier_phase']
reference_end=13.3;stop=np.searchsorted(pt,reference_end)
pt=pt[:stop];pp=pp[:stop]
N=4096;win=signal.windows.hann(N,sym=False).astype(np.float32)
positive=np.flatnonzero((fft.fftfreq(N,1/fs)>=40000)&(fft.fftfreq(N,1/fs)<=850000))
negative=(-positive)%N
names=['receiver_zero','polynomial_moving_line','phase_tracked_moving_line','phase_tracked_line_plus18k','fixed_axis_minus174k']
cross={k:[] for k in names};power={k:[] for k in names};times=[]
L=fs//5
for block,pos in enumerate(range(0,len(raw)-L+1,L)):
    a=raw[pos:pos+L].astype(np.float32)/32768
    original=a[:,0]+1j*a[:,1];t=(pos+np.arange(L))/fs
    polynomial=2*np.pi*(coeff[0]*t**3/3+coeff[1]*t*t/2+coeff[2]*t)
    tracked=polynomial+np.interp(t,pt,pp)
    phases=[None,polynomial,tracked,tracked+2*np.pi*18000*t,2*np.pi*(-174000)*t]
    for name,phase in zip(names,phases):
        value=original if phase is None else (original*np.exp(-1j*phase)).astype(np.complex64)
        count=len(value)//N
        x=fft.fft(value[:count*N].reshape(count,N)*win,axis=1)
        left=x[:,negative];right=x[:,positive]
        cross[name].append(np.mean(left*right,axis=0))
        power[name].append(np.array([np.mean(abs(left)**2,axis=0),np.mean(abs(right)**2,axis=0)]))
    times.append((pos+L/2)/fs)
times=np.array(times)
seed=20261125;rng=np.random.default_rng(seed)
pre=np.flatnonzero((times>=.5)&(times<=12.5));perm=rng.permutation(pre)
train=perm[:len(perm)//2];test=perm[len(perm)//2:];post=np.flatnonzero((times>=13.9)&(times<=15.5))
# Same block phase controls across hypotheses, retaining all within-block bins.
control_phases=rng.uniform(-np.pi,np.pi,(400,len(test)))
post_phases=rng.uniform(-np.pi,np.pi,(400,len(post)))
results=[];arrays={'offset_frequency_hz':positive*fs/N,'time_s':times}
for name in names:
    c=np.array(cross[name]);p=np.array(power[name]);arrays[name+'_cross']=c.astype(np.complex64);arrays[name+'_power']=p.astype(np.float32)
    scale=np.sqrt(p[train,0].mean(axis=0)*p[train,1].mean(axis=0))
    normalized=c/np.maximum(scale,1e-30)
    fitted=normalized[train].mean(axis=0)
    def matched_score(mean):
        # Signed inner product rewards agreement in both shape and phase.
        return float(np.vdot(fitted,mean).real/np.sqrt(np.vdot(fitted,fitted).real*np.vdot(mean,mean).real))
    measured=matched_score(normalized[test].mean(axis=0))
    controls=np.array([matched_score((normalized[test]*np.exp(2j*phase[:,None])).mean(axis=0)) for phase in control_phases])
    after=matched_score(normalized[post].mean(axis=0))
    postcontrols=np.array([matched_score((normalized[post]*np.exp(2j*phase[:,None])).mean(axis=0)) for phase in post_phases])
    # Raw pooled circularity integrates across offsets; it can miss phase slope.
    def pooled(ids):
        numerator=abs(c[ids].mean(axis=0).sum())
        denominator=np.sqrt(p[ids,0].mean(axis=0).sum()*p[ids,1].mean(axis=0).sum())
        return float(numerator/denominator)
    results.append({'hypothesis':name,'heldout_phase_shape_score':measured,
                    'phase_control_95pct':float(np.quantile(controls,.95)),
                    'phase_control_max':float(controls.max()),
                    'controls_exceeding':int(np.sum(controls>=measured)),
                    'control_count':len(controls),
                    'post_event_phase_shape_score':after,
                    'post_control_95pct':float(np.quantile(postcontrols,.95)),
                    'pre_event_pooled_circularity':pooled(pre),'post_event_pooled_circularity':pooled(post)})

# Numerical controls exercise the same conjugate-bin statistic. They do not
# estimate real-recording signal sensitivity or phase-tracking error.
def circularity(samples):
    x=fft.fft(samples*win,axis=1)
    a=x[:,negative];b=x[:,positive]
    return float(abs(np.mean(a*b,axis=0).sum())/np.sqrt(np.mean(abs(a)**2,axis=0).sum()*np.mean(abs(b)**2,axis=0).sum()))
noise=(rng.normal(size=(500,N))+1j*rng.normal(size=(500,N))).astype(np.complex64)/np.sqrt(2)
checks={'real_amplitude_waveform':circularity(noise.real.astype(np.complex64)),
        'proper_complex_noise':circularity(noise),
        'iq_image_epsilon_0p1':circularity(noise+.1*noise.conj()),
        'iq_image_expected':2*.1/(1+.1**2)}
assert checks['real_amplitude_waveform']>.999
assert checks['proper_complex_noise']<.01
assert abs(checks['iq_image_epsilon_0p1']-checks['iq_image_expected'])<.01
report={'scope':'Offline original WAV; no acquisition or transmission.',
        'method':'Hann STFT conjugate-bin covariance E[X(+f)X(-f)] about each coordinate hypothesis. Train-defined frequency profile, disjoint whole 200ms heldout blocks, 400 block phase randomizations.',
        'offset_range_hz':[40000,850000],'seed':seed,'training_blocks':train.tolist(),'heldout_blocks':test.tolist(),'post_event_blocks':post.tolist(),
        'phase_tracker_source':'Previously saved moving-line tracker. Narrow line only; sideband region excluded. Phase tracking frozen at 13.3s for the post-event control.',
        'results':results,'numerical_controls':checks,
        'limitations':['Power symmetry need not imply conjugate phase coupling.','No detected coupling does not exclude weak AM/BPSK mixed with other signals or frequency-selective channels.','A positive receiver-zero result may arise from receiver I/Q imbalance, not an over-air modulation.','Coordinate and phase-tracker choices derive from earlier whole-record measurements, so this split tests covariance stability rather than an end-to-end blind receiver.']}
np.savez_compressed(artifact_path('sideband_phase_coupling_arrays.npz'),**arrays,control_phases=control_phases,post_control_phases=post_phases)
(artifact_path('sideband_phase_coupling.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({'results':results,'numerical_controls':checks},indent=2),flush=True)
