"""Known-input closure controls for the new raw-sample decoder."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E003')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import ast,json
import numpy as np
from scipy.optimize import minimize_scalar
import audit_nonpilot_leakage as core

OUT=experiment_output('E003', '')
tree=ast.parse((artifact_path('projected_nonpilot_decoder.py')).read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
namespace={'np':np,'core':core,'symbol':np.arange(2,302),'fs':core.fs,'CP':2/15*1e-6,'TS':4.4e-6,'minimize_scalar':minimize_scalar}
exec(compile(ast.Module(body=functions,type_ignores=[]),'projected_decoder_functions','exec'),namespace)
decode=namespace['frame_decode']
rng=np.random.default_rng(20261120)
anchor=core.meta['windows'][7]['winner'];fs=core.fs;n=np.arange(50000);t=n/fs;epoch=(anchor['integer_epoch_sample']+anchor['fractional_epoch_offset_samples'])/fs
frameid=np.floor((t-epoch)*750).astype(int);within=t-epoch-frameid/750;sym=np.floor(within/4.4e-6).astype(int)
active=(sym>=2)&(sym<302);ids=np.arange(520,545);freq=(ids-531.5)*234375;valid=abs(freq+anchor['fractional_tracking_cfo_hz'])<fs*.49;ids=ids[valid];freq=freq[valid]
gains=np.exp(1j*rng.uniform(-np.pi,np.pi,len(ids)));codes=rng.choice([-1,1],size=(17,60))
def generate(with_nonpilot):
    values=np.zeros(len(n),complex)
    for j,k in enumerate(ids):
        if 528<=k<=535:a=core.expected[sym[active]-2,k-528]
        elif with_nonpilot:
            ref=np.exp(.5j*np.pi*core.rot[k,sym[active]-1]);a=ref*codes[frameid[active]+1,(k-16*sym[active])%60]
        else:continue
        values[active]+=gains[j]*a*np.exp(2j*np.pi*freq[j]*(within[active]-sym[active]*4.4e-6-2/15*1e-6))/np.sqrt(8)
    return values*np.exp(2j*np.pi*anchor['fractional_tracking_cfo_hz']*t)

positive=generate(True);null=generate(False);errors=[];correct=0;total=0;nullpowers=[]
for frame in [1,4,8,12]:
    result=decode(positive,anchor,frame);blank=decode(null,anchor,frame)
    for col,k in enumerate(result['unknown']):
        gain=gains[np.where(ids==k)[0][0]];reference=np.exp(.5j*np.pi*core.rot[k,core.symbol-1]);truth=reference*codes[frame+1,(k-16*core.symbol)%60]
        recovered=result['values'][:,col]/gain
        # A single fitted carrier phase is a nuisance; eliminate it for numerical closure.
        phase=np.angle(np.vdot(truth,recovered));recovered*=np.exp(-1j*phase)
        errors.append(float(np.sqrt(np.mean(abs(recovered-truth)**2))));sign=(recovered*reference.conj()).real>0;target=codes[frame+1,(k-16*core.symbol)%60]>0
        correct+=int(np.sum(sign==target));total+=len(target)
    nullpowers.append(float(np.mean(abs(blank['values'])**2)))
report={'random_seed':20261120,'source':'Synthetic known pilots plus known per-frame BPSK/15-symbol nonpilot pattern, using the same raw sampling geometry. This is a numerical control, not extra real RF data.','frames':[1,4,8,12],'available_nonpilot_subcarriers':result['unknown'].tolist(),'positive_sign_agreement':correct/total,'positive_rms_error_max':max(errors),'pilot_only_null_estimated_nonpilot_power_max':max(nullpowers),'absolute_polarity_validation':False,'actual_message_decoded':False}
(artifact_path('projected_decoder_closure_controls.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2),flush=True)
assert report['positive_sign_agreement']>.999
assert report['positive_rms_error_max']<.01
assert report['pilot_only_null_estimated_nonpilot_power_max']<1e-5
print('PASS: known-input symbol closure and pilot-only rejection.',flush=True)
