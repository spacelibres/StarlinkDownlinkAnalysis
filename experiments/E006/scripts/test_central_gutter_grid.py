"""Conditional OFDM central-gutter spectral model, tested on held-out PSD rows."""

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
import json
import numpy as np
from scipy import ndimage

OUT=experiment_output('E006', '')
with np.load(artifact_path('waterfall_detail_data.npz'),allow_pickle=False) as z:
    freq=z['frequency_hz'];time=z['time_s'];power=z['psd']
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
split=json.loads((artifact_path('global_spectral_axis_validation.json')).read_text(encoding='utf8'))
cal=np.array(split['training_whole_stft_block_indices']);test=np.array(split['heldout_whole_stft_block_indices'])
noise=power[(time>=13.9)&(time<=15.5)].mean(axis=0)
offset=np.arange(-950000,1300001,2500)
keep=(abs(offset)>30000)&~((offset>150000)&(offset<240000))&~((offset>-460000)&(offset<-370000))
offset=offset[keep];rows=[]
for i in np.r_[cal,test]:
    sample=np.polyval(coeff,time[i])+offset
    before=ndimage.gaussian_filter1d(power[i],100)
    floor=ndimage.gaussian_filter1d(noise,100)
    rows.append(np.interp(sample,freq,before-floor)/np.maximum(np.interp(sample,freq,floor),1e-30))
rows=np.array(rows);c=rows[:len(cal)].mean(axis=0);v=rows[len(cal):].mean(axis=0)
spacing=234375.;k=np.arange(-20,20);records=[];arrays={'offset_from_moving_line_hz':offset,'calibration_excess_ratio':c,'heldout_excess_ratio':v}
for empty in [0,2,4,6,8]:
    selected=[]
    for shift in np.arange(-150000,150001,1000):
        loaded=k[(k<-empty/2)|(k>=empty/2)] if empty else k
        centres=(loaded+.5)*spacing+shift
        basis=np.sinc((offset[:,None]-centres[None,:])*4.4e-6)**2
        design=np.column_stack([basis.sum(axis=1),basis@(centres/1e6),np.ones(len(offset))])
        beta=np.linalg.lstsq(design,c,rcond=None)[0]
        if beta[0]<=0 or abs(beta[1])*1.35>beta[0]:continue
        fit=design@beta;error=float(np.mean((c-fit)**2))
        selected.append((error,float(shift),beta,fit))
    winner=min(selected,key=lambda item:item[0]);fit=winner[3]
    correlation=float(np.corrcoef(fit,v)[0,1])
    row={'central_null_count_hypothesis':empty,'gutter_midpoint_offset_from_moving_line_hz':winner[1],
         'calibration_mse':winner[0],'heldout_mse_with_frozen_calibration_parameters':float(np.mean((v-fit)**2)),
         'heldout_shape_correlation':correlation,'calibration_coefficients':winner[2].tolist()}
    records.append(row);arrays['nulls_'+str(empty)+'_prediction']=fit
best=min(records,key=lambda row:row['calibration_mse'])
result={'scope':'Conditional spectral-shape test, not unknown-symbol decoding.',
        'subcarrier_spacing_hz_fixed_from_public_hypothesis':spacing,'slot_duration_s':4.4e-6,
        'relative_grid':'f = moving_line + midpoint_offset + (signed_k+0.5)*spacing',
        'calibration_blocks':cal.tolist(),'heldout_blocks':test.tolist(),'models':records,'calibration_selected_model':best,
        'limitations':['Carrier powers use only an affine spectral tilt plus a constant baseline.','Assumes locally independent OFDM symbols with approximate rectangular slot spectra.','Post-event spectrum is a proxy for receiver/background response, not certified thermal noise.','Null count and spacing are tested hypotheses; a shape fit is not proof of a particular protocol or channel.','Frequency-grid midpoint differs from the k=0 OFDM carrier by half a subcarrier spacing.']}
(artifact_path('central_gutter_grid_test.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
np.savez_compressed(artifact_path('central_gutter_grid_test_arrays.npz'),**arrays)
print(json.dumps({'models':records,'selected':best},indent=2))
