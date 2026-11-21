"""Combine raw/dechirped test families and check the lag-chirp prediction."""

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

OUT=experiment_output('E005', '')
raw=json.loads((artifact_path('subband_cyclostationarity.json')).read_text(encoding='utf8'))
corrected=json.loads((artifact_path('subband_cyclostationarity_dechirped.json')).read_text(encoding='utf8'))
with np.load(artifact_path('subband_cyclostationarity_arrays.npz'),allow_pickle=False) as a,np.load(artifact_path('subband_cyclostationarity_dechirped_arrays.npz'),allow_pickle=False) as b:
    assert np.array_equal(a['heldout_random_phases'],b['heldout_random_phases'])
    joint=np.maximum(a['joint_max_null'],b['joint_max_null'])
rows=[]
for name,data in [('raw',raw),('dechirped',corrected)]:
    for row in data['candidates']:
        output=dict(row);output['coordinate_run']=name
        output['joint_36_family_p']=float((1+np.sum(joint>=row['heldout_coherent_score']))/1000)
        rows.append(output)
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
times=(np.array(raw['calibration_blocks'])+.5)*.2
rate=float(np.mean(2*coeff[0]*times+coeff[1]));drift=[]
for r in raw['candidates']:
    if r['selection_kind']!='blind_coarse_search' or r['band']=='center_control':continue
    match=next(c for c in corrected['candidates'] if c['band']==r['band'] and c['coarse_cycle_hz']==r['coarse_cycle_hz'] and c['selection_kind']=='blind_coarse_search')
    lag=r['lag_samples_at400k']/400000
    drift.append({'band':r['band'],'harmonic_cycle_hz':r['coarse_cycle_hz'],'lag_s':lag,
                  'observed_raw_offset_hz':r['cycle_hz']-r['coarse_cycle_hz'],
                  'predicted_from_line_rate_offset_hz':lag*rate,
                  'dechirped_offset_hz':match['cycle_hz']-match['coarse_cycle_hz']})
consistency={'prior_line_mean_frequency_rate_hz_per_s':rate,'relation':'alpha_observed approximately alpha_intrinsic + lag_seconds * frequency_rate_hz_per_s','rows':drift,
             'max_abs_raw_prediction_error_hz':max(abs(x['observed_raw_offset_hz']-x['predicted_from_line_rate_offset_hz']) for x in drift),
             'max_abs_dechirped_harmonic_offset_hz':max(abs(x['dechirped_offset_hz']) for x in drift),
             'caution':'Earlier whole-record trajectory is the correction reference. Grid spacing 0.05Hz is not calibrated accuracy or a frequency confidence interval.'}
(artifact_path('cyclic_drift_consistency.json')).write_text(json.dumps(consistency,indent=2),encoding='utf8')
summary={'tested_coordinates':['raw','dechirped'],'joint_candidate_family_size':len(rows),'control_count':len(joint),
         'joint_36_max_null_95pct':float(np.quantile(joint,.95)),'candidates':rows,
         'by_band':{},'interpretation':'Shared cyclic timing around 750Hz and its harmonics, with lag-dependent raw offsets consistent with the moving-line chirp. Does not identify symbol rate, OFDM cyclic prefix, dual polarization, source hardware or message bits.',
         'statistical_scope':'Conditional max-statistic random-phase inference over 36 frozen tests; raw/dechirped runs use identical control phases. Prior 750Hz hypotheses remain explicitly labeled, not blind discoveries.'}
for band in raw['bands_hz']:
    chosen=[r for r in rows if r['band']==band]
    summary['by_band'][band]={'blind_tests':sum(r['selection_kind']=='blind_coarse_search' for r in chosen),
                            'blind_passes_joint_36':sum(r['selection_kind']=='blind_coarse_search' and r['joint_36_family_p']<=.05 for r in chosen),
                            'seeded_750Hz_joint_p_by_run':{r['coordinate_run']:r['joint_36_family_p'] for r in chosen if r['selection_kind']=='historical_750Hz_prior'}}
(artifact_path('cyclic_evidence_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({'by_band':summary['by_band'],'joint_36_threshold':summary['joint_36_max_null_95pct'],'drift_prediction_max_error_hz':consistency['max_abs_raw_prediction_error_hz'],'dechirped_offset_max_hz':consistency['max_abs_dechirped_harmonic_offset_hz']},indent=2))
