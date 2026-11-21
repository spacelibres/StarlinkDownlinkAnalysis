"""Filter-aware comparison of the single local termination event.

This is one-event parameter estimation, not an independent event replication.
Noise resampling retains simultaneous bands in whole 20-ms groups.
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
from scipy import signal
from scipy.io import wavfile

OUT=experiment_output('E006', '')
parser=argparse.ArgumentParser()
parser.add_argument('--bin-ms',type=float,default=1)
parser.add_argument('--group-ms',type=float,default=20)
parser.add_argument('--fit-start',type=float,default=13.35)
parser.add_argument('--fit-end',type=float,default=13.95)
parser.add_argument('--bootstrap',type=int,default=400)
parser.add_argument('--suffix',default='')
args=parser.parse_args();bin_s=args.bin_ms/1000
fs,raw=wavfile.read(RAW_WAV,mmap=True)
start=12.7;end=14.5;t0=13.635
a=raw[round(start*fs):round(end*fs)].astype(np.float32)/32768
z=a[:,0]+1j*a[:,1];time=start+np.arange(len(z))/fs
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
phase=2*np.pi*(coeff[0]*time**3/3+coeff[1]*time*time/2+coeff[2]*time)
low=signal.resample_poly((z*np.exp(-1j*phase)).astype(np.complex64),1,120)
lf=fs/120;lt=start+np.arange(len(low))/lf
unit=signal.resample_poly((time<t0).astype(np.float32),1,120)
profiles={};templates={};specs={};win=round(fs*bin_s)
bt=start+(np.arange(len(z)//win)+.5)*bin_s

def average_full(power):return power[:len(bt)*win].reshape(-1,win).mean(axis=1)
def average_low(power):
    count=round(lf*bin_s)
    return power[:len(bt)*count].reshape(-1,count).mean(axis=1)

# Centered symmetric FIR convolution removes its known linear delay, retaining
# the finite edge spread. Guard data are much longer than all impulse responses.
for name,lo,hi in [('negative_lobe',-950000,-650000),('positive_lobe',500000,750000),('center_control',-100000,100000)]:
    h=signal.firwin(513,(hi-lo)/2,fs=fs,window=('kaiser',8)).astype(np.float32)
    shifted=(z*np.exp(-2j*np.pi*(lo+hi)/2*time)).astype(np.complex64)
    filtered=signal.fftconvolve(shifted,h,mode='same')
    profiles[name]=average_full(abs(filtered)**2)
    # White-in-passband power step: tail integral of |h|^2.
    tau=(np.arange(len(h))-(len(h)-1)/2)/fs
    tail=np.cumsum((h*h)[::-1])[::-1];tail/=tail[0]
    templates[name]=(tau,tail)
    specs[name]={'band_hz':[lo,hi],'fir_taps':513,'causal_group_delay_s':256/fs,'delay_removed':True,'power_step_model':'white-in-passband tail energy','response_support_s':512/fs}

for cutoff in [125,250,500]:
    name=f'moving_line_lp{cutoff}'
    h=signal.firwin(1201,cutoff,fs=lf,window=('kaiser',8)).astype(np.float32)
    filtered=signal.fftconvolve(low,h,mode='same')
    profiles[name]=average_low(abs(filtered)**2)
    # End-to-end deterministic gated tone through decimator and identical FIR.
    response=abs(signal.fftconvolve(unit,h,mode='same'))**2
    select=abs(lt-t0)<.08
    templates[name]=(lt[select]-t0,response[select])
    specs[name]={'lowpass_cutoff_hz':cutoff,'post_decimation_rate_hz':lf,'fir_taps':1201,'causal_group_delay_s':600/lf,'delay_removed':True,'power_step_model':'coherent gated tone through the same decimator and FIR','response_support_s':1200/lf,'residual_phase_variation_not_in_model':True}

fitmask=(bt>=args.fit_start)&(bt<args.fit_end);fit_time=bt[fitmask]
candidate=np.arange(13.55,13.720001,.0001)
trend=np.stack([np.ones(len(fit_time)),fit_time-t0],axis=1)
projection=np.linalg.pinv(trend)
sample_offsets=(np.arange(20)+.5)/20*bin_s-bin_s/2
models={};results=[];fits={};residuals={}
for name,power in profiles.items():
    tx,ty=templates[name]
    response=np.empty((len(candidate),len(fit_time)))
    for left in range(0,len(candidate),64):
        response[left:left+64]=np.mean(np.interp(fit_time[None,:,None]+sample_offsets[None,None,:]-candidate[left:left+64,None,None],tx,ty,left=1,right=0),axis=-1)
    # Remove constant and linear trend from each candidate template analytically.
    residual=response-(response@projection.T)@trend.T
    norm=np.sqrt(np.sum(residual**2,axis=1));normalized=residual/np.maximum(norm[:,None],1e-20)
    models[name]=normalized
    y=power[fitmask]
    score=normalized@y
    k=int(np.argmax(score));edge=float(candidate[k])
    design=np.column_stack([trend,response[k]])
    parameters=np.linalg.lstsq(design,y,rcond=None)[0]
    fitted=design@parameters
    fits[name]=fitted;residuals[name]=y-fitted
    before=float(parameters[0]+parameters[2]);after=float(parameters[0])
    results.append({'component':name,'edge_time_s':edge,'pre_minus_post_power':float(parameters[2]),
                    'local_drop_db':float(10*np.log10(max(after,1e-30)/max(before,1e-30))),
                    'edge_grid_at_boundary':bool(k in [0,len(candidate)-1]),'filter':specs[name]})

# The transition is not withheld. Pre/post here are baseline noise strata, not
# train/test sets. Draw aligned vectors of residuals across all six components.
names=list(profiles);R=np.column_stack([residuals[k] for k in names])
group=round(args.group_ms/args.bin_ms);groups=[];strata=[]
for i in range(0,len(fit_time)-group+1,group):
    ts=fit_time[i:i+group]
    if ts[-1]<t0-.06:groups.append(R[i:i+group]);strata.append('pre')
    elif ts[0]>t0+.06:groups.append(R[i:i+group]);strata.append('post')
groups=np.array(groups);strata=np.array(strata)
seed=20261127;rng=np.random.default_rng(seed);B=args.bootstrap
choices=np.zeros((B,int(np.ceil(len(fit_time)/group))),dtype=int)
for block in range(choices.shape[1]):
    pool=np.flatnonzero(strata==('pre' if fit_time[min(block*group,len(fit_time)-1)]<t0 else 'post'))
    choices[:,block]=rng.choice(pool,size=B)
simnoise=groups[choices].reshape(B,-1,len(names))[:,:len(fit_time)]
simnoise-=simnoise.mean(axis=1,keepdims=True)
boots={}
for j,name in enumerate(names):
    simulated=fits[name][None,:]+simnoise[:,:,j]
    score=simulated@models[name].T
    edges=candidate[np.argmax(score,axis=1)]
    boots[name]=edges
    row=next(r for r in results if r['component']==name)
    row['conditional_noise_bootstrap_95pct_s']=np.quantile(edges,[.025,.975]).tolist()
    row['bootstrap_boundary_fraction']=float(np.mean((edges==candidate[0])|(edges==candidate[-1])))
    row['delay_vs_negative_lobe_s']=row['edge_time_s']-results[0]['edge_time_s']
    row['conditional_delay_bootstrap_95pct_s']=np.quantile(edges-boots['negative_lobe'],[.025,.975]).tolist()

report={'scope':'Existing original WAV only, no acquisition/transmission.',
        'event_count':1,'power_bin_s':bin_s,'edge_search_grid_s':.0001,'fitting_interval_s':[args.fit_start,args.fit_end],
        'candidate_interval_s':[13.55,13.72],'seed':seed,'bootstrap_replicates':B,'noise_group_duration_s':args.group_ms/1000,
        'noise_strata_group_indices':{'pre':np.flatnonzero(strata=='pre').tolist(),'post':np.flatnonzero(strata=='post').tolist()},
        'results':results,
        'limitations':['Only one observed termination; bootstrap replicates are not additional events.','Intervals condition on an abrupt step model plus common linear power trend and the earlier moving-line frequency model.','White-in-passband broad-signal and locally coherent tone responses are approximations.','Known FIR delays are removed; edge spreading and unknown receiver front-end delays remain.','Search grid and sample spacing do not determine statistical timing precision.','A shared edge would support association, not uniquely identify a transmitter, clock, or beam.'],
        'references':{'filter_group_delay':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html'}}
prefix='burst_edge_alignment'+('_'+args.suffix if args.suffix else '')
(artifact_path(prefix+'.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
np.savez_compressed(artifact_path(prefix+'_arrays.npz'),time_s=bt,fit_time_s=fit_time,candidate_edge_time_s=candidate,noise_group_choices=choices,
    **{k+'_power':v for k,v in profiles.items()},**{k+'_fit':v for k,v in fits.items()},**{k+'_bootstrap_edges':v for k,v in boots.items()})
print(json.dumps({'results':results},indent=2),flush=True)
