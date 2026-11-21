"""Known non-grid cycle and stationary null; not extra RF observations."""

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
import json
import numpy as np
from analyze_subband_cyclostationarity import refine,scf_profile,CF,RF,BLOCK,CLEN,LEN,LAGS
from scipy import signal as dsp

OUT=experiment_output('E005', '');rng=np.random.default_rng(20261129)
count=65;perm=rng.permutation(count);train=perm[:32];test=perm[32:];truth=1234.3
t=(np.arange(count)[:,None]*CLEN+np.arange(CLEN)[None,:])/CF
noise=(rng.normal(size=t.shape)+1j*rng.normal(size=t.shape))/np.sqrt(2)
power=abs(noise)**2*(1+.6*np.cos(2*np.pi*truth*t))
products=np.zeros((count,len(LAGS),CLEN),np.complex64)
products[:,0]=power/power.mean(axis=1,keepdims=True)-1
frequency,v,score=refine(products,train,1235,0)
def statistic(x):return float(abs(x.sum())**2/np.sum(abs(x)**2))
real=statistic(v[test]);nullvalues=abs(noise)**2
products[:,0]=nullvalues/nullvalues.mean(axis=1,keepdims=True)-1
nf,nv,ns=refine(products,train,1235,0);null=statistic(nv[test])
phases=rng.uniform(-np.pi,np.pi,(999,len(test)))
control=abs(np.exp(1j*phases)@v[test])**2/np.sum(abs(v[test])**2)

blocks=8;time=(np.arange(blocks)[:,None]*LEN+np.arange(LEN)[None,:])/RF
base=(rng.normal(size=time.shape)+1j*rng.normal(size=time.shape)).astype(np.complex64)/np.sqrt(2)
signal=base*np.sqrt(1+.6*np.cos(2*np.pi*truth*time))
raw,coherent=scf_profile(signal,truth,np.arange(blocks))
wrong,wc=scf_profile(signal,1700,np.arange(blocks))
zero,zc=scf_profile(signal,0,np.arange(blocks))
result={'seed':20261129,'kind':'Synthetic periodically varying variance of proper complex noise; checks non-grid phase/time convention, cycle localization, and SCF estimator. Not a measured symbol-rate example.',
        'known_cycle_hz':truth,'recovered_cycle_hz':frequency,'positive_heldout_score':real,
        'positive_phase_control_99pct':float(np.quantile(control,.99)),
        'stationary_null_heldout_score':null,
        'target_scf_mean_real_coherence':float(np.mean(coherent.real)),
        'wrong_cycle_mean_abs_coherence':float(np.mean(abs(wc))),
        'zero_cycle_coherence_max_error_from_one':float(np.max(abs(zc-1)))}
# Check lag padding, centering and product decimation on stationary colored
# complex noise. A false 750Hz cycle must not be manufactured by block edges.
white=(rng.normal(size=count*LEN+1024)+1j*rng.normal(size=count*LEN+1024)).astype(np.complex64)/np.sqrt(2)
colored=dsp.fftconvolve(white,dsp.firwin(129,120000,fs=RF),mode='same')[512:512+count*LEN].reshape(count,LEN)
window=dsp.windows.hann(CLEN,sym=False)
kernel=window*np.exp(-2j*np.pi*750*np.arange(CLEN)/CF)/window.sum()
nullcoeff=np.empty((count,len(LAGS)),complex)
for b in range(count):
    x=colored[b];power=np.mean(abs(x)**2)
    for j,lag in enumerate(LAGS):
        product=np.zeros(LEN,np.complex64)
        if lag:product[:-lag]=x[lag:]*x[:-lag].conj()
        else:product=x*x.conj()
        product-=product[:LEN-lag if lag else LEN].mean()
        down=dsp.resample_poly(product,1,20,window=('kaiser',8))[:CLEN]/power
        nullcoeff[b,j]=np.dot(down,kernel)*np.exp(-2j*np.pi*750*(b*BLOCK+lag/(2*RF)))
cal=abs(nullcoeff[train].sum(axis=0))**2/np.sum(abs(nullcoeff[train])**2,axis=0)
chosen=int(np.argmax(cal));held=statistic(nullcoeff[test,chosen])
result['stationary_colored_noise_750Hz_train_selected_lag']=int(LAGS[chosen])
result['stationary_colored_noise_750Hz_heldout_score']=held
assert abs(frequency-truth)<.11
assert real>np.quantile(control,.99) and real>15 and null<10
assert result['target_scf_mean_real_coherence']>.2
assert result['wrong_cycle_mean_abs_coherence']<.1
assert result['zero_cycle_coherence_max_error_from_one']<1e-6
assert held<10
(artifact_path('cyclic_estimator_controls.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
