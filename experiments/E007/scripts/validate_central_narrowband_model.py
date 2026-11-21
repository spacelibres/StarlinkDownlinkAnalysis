"""Forward full-band OFDM -> narrowband receiver -> local estimator control.

Known synthetic symbols, not an actual extra Starlink capture. Tests whether
front-end filtering invalidates an otherwise well-conditioned tone fit.
"""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E007')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import ast,json
import numpy as np
from scipy import fft,signal
from scipy.io import loadmat

OUT=experiment_output('E007', '')
tree=ast.parse((artifact_path('extract_central_ofdm_candidates.py')).read_text(encoding='utf8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='demod_frame')
namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),'central-demod-function','exec'),namespace)
demod=namespace['demod_frame'];rng=np.random.default_rng(20261203)
pn=np.zeros(32767,dtype=int);pn[:15]=[int(x) for x in '000000100000010']
for i in range(15,len(pn)):pn[i]=pn[i-14]^pn[i-15]
k=np.arange(1024);signed=np.where(k<512,k,k-1024);ordinal=np.where(signed<0,signed+512,signed+508)
s=np.arange(2,302);d=1020*(s[:,None]-17)+ordinal[None,:]
rotation=(2*pn[(d+16327)%32767]+(1^pn[(d+16320)%32767])+1)%4
reference=np.exp(.5j*np.pi*rotation)
bits=rng.choice([-1,1],60);information=bits[(k[None,:]-16*s[:,None])%60]
data=reference*information;data[:,[0,1,1022,1023]]=0
sss=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel()
body=32*fft.ifft(np.vstack([sss,data]),axis=1)
body=np.concatenate([body[:,-32:],body],axis=1).reshape(-1)
reg=[0,0,1,1,0,1,0];seq=[]
for _ in range(127):v=reg[2]^reg[6];reg=[v]+reg[:-1];seq.append(v)
segment=np.exp(-1j*np.pi/4-.5j*np.pi*np.cumsum(2*np.r_[0,seq[::-1]]-1))
pss=np.r_[-segment,np.tile(segment,7)];pss=np.r_[-pss[-32:],pss]
wave=np.r_[pss,body,np.zeros(1088)]
assert len(wave)==320000
fullrate=240000000.;center=-180000+117187.5
t=np.arange(len(wave))/fullrate
captured=signal.resample_poly(wave*np.exp(2j*np.pi*center*t),1,100,window=('kaiser',8))
padding=2400;epoch=padding/2400000
receiver=np.r_[np.zeros(padding,complex),captured,np.zeros(padding,complex)]
receiver*=np.exp(2j*np.pi*center*epoch)
raw=np.column_stack([receiver.real,receiver.imag])*32768
selected=np.array([-4,-3,2,3,4,5]);indices=selected%1024
estimates,error,condition=demod(raw,epoch,2400000,1.,np.array([0.,0.,-180000.]),(selected+.5)*234375)
truth=data[:,indices];cal=np.arange(0,300,2);test=np.arange(1,300,2)
gain=np.sum(estimates[1:][cal]*truth[cal].conj(),axis=0)/np.sum(abs(truth[cal])**2,axis=0)
recovered=estimates[1:]/gain[None,:]
agreement=(np.sign((recovered*reference[:,indices].conj()).real)==information[:,indices])
result={'kind':'Synthetic full 240MHz-grid OFDM with known PN/BPSK data, anti-alias filtered and decimated to 2.4MS/s. No real RF added.',
        'seed':20261203,'subcarriers_signed':selected.tolist(),'noiseless_heldout_sign_agreement_by_carrier':agreement[test].mean(axis=0).tolist(),
        'noiseless_heldout_rms_by_carrier':np.sqrt(np.mean(abs(recovered[test]-truth[test])**2,axis=0)).tolist(),
        'median_design_condition':condition,'rx_filter_distortion_present':True,
        'caution':'Channel calibration uses known synthetic even-slot data; this is an optimistic model-control, not an actual payload decoder or measured BER.'}
(artifact_path('central_narrowband_model_control.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
assert np.all(np.array(result['noiseless_heldout_sign_agreement_by_carrier'])[np.isin(selected,[-3,2,3,4])]>.90)
