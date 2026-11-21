"""Does the broad, time-varying spectrum sharpen when the thin-line drift is removed?"""

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
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
split=json.loads((artifact_path('global_spectral_axis_validation.json')).read_text(encoding='utf8'))
train=split['training_whole_stft_block_indices'];test=split['heldout_whole_stft_block_indices']
ids=np.array(train+test);trainmask=np.isin(ids,train);testmask=~trainmask
post=p[(t>=13.9)&(t<=15.5)].mean(axis=0)
grid=np.arange(-1000000,1000001,2000)
center_time=float(np.mean(t[train]));center_freq=float(np.polyval(coeff,center_time))
shift=np.polyval(coeff,t)-center_freq
# Mask the entire moving-line corridor and fixed-line corridors, under either
# coordinate transformation. This test is about broad spectrum, not line sharpening.
keep=(abs(grid-center_freq)>90000)&(abs(grid-13000)>50000)&(abs(grid+600000)>50000)
models={'receiver_coordinates':[],'moving_line_drift_removed':[],'opposite_drift_control':[]}
for i in ids:
    excess=ndimage.gaussian_filter1d(p[i]-post,100)
    for name,offset in [('receiver_coordinates',0),('moving_line_drift_removed',shift[i]),('opposite_drift_control',-shift[i])]:
        value=np.interp(grid+offset,f,excess)[keep]
        value-=value.mean()
        models[name].append(value/np.sqrt(np.dot(value,value)))
correlations={};profiles={}
for name,values in models.items():
    values=np.array(values);profile=values[trainmask].mean(axis=0);profile/=np.linalg.norm(profile)
    profiles[name]=profile;correlations[name]=values[testmask]@profile
delta=correlations['moving_line_drift_removed']-correlations['receiver_coordinates']
seed=20261126;rng=np.random.default_rng(seed)
boot=rng.choice(delta,size=(4000,len(delta)),replace=True).mean(axis=1)
result={'method':'Held-out normalized broad spectral-shape correlation against train-only mean profile, with all narrow-line corridors excluded. Same independent STFT block assignments as spectral-axis validation.',
        'training_blocks':train,'heldout_blocks':test,'bootstrap_seed':seed,
        'coordinate_reference_time_s':center_time,'coordinate_reference_moving_frequency_hz':center_freq,
        'applied_shift_range_hz':[float(shift[ids].min()),float(shift[ids].max())],
        'heldout_mean_correlations':{name:float(c.mean()) for name,c in correlations.items()},
        'paired_improvement_drift_removed_minus_fixed':float(delta.mean()),
        'bootstrap_95pct_improvement':np.quantile(boot,[.025,.975]).tolist(),
        'caution':'Magnitude spectra only, smoothed at 3.66kHz; common post-event baseline. Whole-record thin-line trajectory is an existing descriptive reference, not a blind estimate on these splits. A failure cannot exclude a shared source with frequency-selective response.'}
(artifact_path('wideband_drift_link.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
np.savez_compressed(artifact_path('wideband_drift_link_arrays.npz'),retained_receiver_frequency_hz=grid[keep],heldout_blocks=np.array(test),**{name+'_heldout_correlations':c for name,c in correlations.items()},**{name+'_training_profile':c for name,c in profiles.items()})
print(json.dumps({k:v for k,v in result.items() if k not in ['training_blocks','heldout_blocks']},indent=2))
