"""Validate spectral reflection axes on held-out whole STFT blocks."""

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
from scipy import ndimage

OUT=experiment_output('E004', '')
with np.load(artifact_path('waterfall_detail_data.npz'),allow_pickle=False) as z:
    f=z['frequency_hz'];t=z['time_s'];p=z['psd']
post=p[(t>=13.9)&(t<=15.5)].mean(axis=0)
ids=np.flatnonzero((t>=.5)&(t<=12.5))
seed=20261124;perm=np.random.default_rng(seed).permutation(ids)
train=perm[:len(perm)//2];test=perm[len(perm)//2:]
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
tone=np.polyval(coeff,t)
distance=np.arange(20000,850001,2000)
offsets=np.arange(-100000,100001,2000)
axes=np.arange(-300000,300001,2000)
dynamic=[];fixed=[];local=[]
for i in ids:
    shape=ndimage.gaussian_filter1d(p[i]-post,100)
    def score(center):
        left=np.interp(center-distance,f,shape);right=np.interp(center+distance,f,shape)
        return float(np.corrcoef(left,right)[0,1])
    dynamic.append([score(tone[i]+d) for d in offsets])
    fixed.append([score(c) for c in axes])
    local.append(score(0))
dynamic=np.array(dynamic);fixed=np.array(fixed);local=np.array(local)
cal=np.isin(ids,train);hold=np.isin(ids,test)
di=int(np.argmax(dynamic[cal].mean(axis=0)));fi=int(np.argmax(fixed[cal].mean(axis=0)))
delta=dynamic[hold,di]-local[hold]
rng=np.random.default_rng(seed+1)
boot=np.mean(rng.choice(delta,size=(3000,len(delta)),replace=True),axis=1)
result={'seed':seed,'training_whole_stft_block_indices':train.tolist(),'heldout_whole_stft_block_indices':test.tolist(),
        'frequency_smoothing_sigma_hz':float((f[1]-f[0])*100),'distance_range_from_axis_hz':[20000,850000],
        'selected_dynamic_axis_offset_from_moving_tone_hz':int(offsets[di]),
        'selected_fixed_axis_hz':int(axes[fi]),
        'heldout_mean_shape_correlations':{'moving_tone_plus_selected_offset':float(dynamic[hold,di].mean()),
                                         'selected_fixed_axis':float(fixed[hold,fi].mean()),'receiver_zero':float(local[hold].mean())},
        'heldout_dynamic_minus_receiver_zero_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist(),
        'caution':'A shape-association test. Moving-tone track was fitted earlier across the recording, not refitted on this split. Common post-event baseline is reused; spectral image/shared modulation are unresolved alternatives. No RF center or physical origin is established.'}
(artifact_path('global_spectral_axis_validation.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if 'indices' not in k},indent=2))
