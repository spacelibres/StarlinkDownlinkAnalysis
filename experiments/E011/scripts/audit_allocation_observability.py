"""Compare frozen resource predictions with header-blind signal diagnostics."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context
from layout_core import eligible_first_headers,parse_descriptors
from allocation_core import coherence,data_field,detect_suffix,least_squares_change,predict_end,UNITS,upward_contrast,unit_identifiability

THRESHOLDS=[.35,.40,.45,.50,.55,.60]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def fit_unit_diagnostic(rows,width):
    selected=[r for r in rows if r['primary_signal_contrast']]
    codes=sorted({e['mcs_candidate'] for r in selected for e in r[f'layout{width}']['entries']})
    if not selected:return {'frames':0}
    matrix=np.array([[sum(e['quantity'] for e in r[f'layout{width}']['entries'] if e['mcs_candidate']==code)
                      for code in codes] for r in selected],dtype=float)
    target=np.array([r['signal']['suffix']['0.45']['boundary_resource']-r['second_header_end'] for r in selected])
    result=unit_identifiability(matrix,target)
    result['unconstrained_minimum_norm_units']=dict(zip(map(str,codes),result['unconstrained_minimum_norm_units']))
    result['positive_unit_example']=dict(zip(map(str,codes),result['positive_unit_example']))
    return {'frames':len(selected),'codes':codes,**result,
            'note':'Fitted on these endpoints, not a frozen prediction or independent format test. Positive solutions are examples, not recovered physical codeword lengths.'}


def main():
    require_run_context();out=experiment_output('E011');out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
    source_hash=digest(path)
    if source_hash!='a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf':raise RuntimeError('Capture changed')
    iq=np.load(path,mmap_mode='r',allow_pickle=False);frames=eligible_first_headers(len(iq))
    seed=2026112011;rng=np.random.default_rng(seed);signals={};profiles={};nulls={}
    # All signal-only calculations finish before any header results are loaded.
    for frame in frames:
        z,raw,wrong=data_field(iq,frame);c=coherence(z);c0=coherence(raw);cw=coherence(wrong)
        profiles[f'frame_{frame+1:02d}_pn']=c;profiles[f'frame_{frame+1:02d}_raw']=c0;profiles[f'frame_{frame+1:02d}_wrong_pn']=cw
        suffix={f'{threshold:.2f}':detect_suffix(c,threshold=threshold) for threshold in THRESHOLDS}
        signals[frame]={'suffix':suffix,'least_squares_1024':least_squares_change(c,min_windows=32),
                        'least_squares_256':least_squares_change(c,min_windows=8),
                        'raw_suffix':detect_suffix(c0),'wrong_pn_suffix':detect_suffix(cw),
                        'coherence_median':float(np.median(c)),'raw_coherence_median':float(np.median(c0)),
                        'wrong_pn_coherence_median':float(np.median(cw))}
        nulls[frame]=[detect_suffix(c[rng.permutation(len(c))]) for _ in range(199)]
        print('signal-only frame',frame,'primary',suffix['0.45'],flush=True)
    header_path=ROOT/'runs/v008-complete-header-layout-02/artifacts/experiments/E010/results/full_header_layout.json'
    bit_path=header_path.with_name('full_header_observations.npz')
    headers=json.loads(header_path.read_text(encoding='utf8'));bits=np.load(bit_path,allow_pickle=False)
    rows=[]
    for header in headers['frames']:
        frame=header['legacy_frame_index'];h2=header['second_header'];start=h2['start_resource'];blocks=h2['blocks']
        if not(header['first_header']['crc_ok'] and h2['crc_ok'] and h2['coded_hard_disagreements']==0):
            raise RuntimeError('Unverified source header')
        entries=h2['layout20']['entries'];prediction=predict_end(start,blocks,entries)
        joint=predict_end(start,blocks,entries,joint_rounding=True);end=prediction['end_resource']
        signal=signals[frame];primary=signal['suffix']['0.45'];boundary=primary['boundary_resource']
        contrast=upward_contrast(primary)
        c=profiles[f'frame_{frame+1:02d}_pn'];before=c[max(0,end//32-32):end//32];after=c[(end+31)//32:min(len(c),(end+31)//32+32)]
        sensitivity=[r['boundary_resource'] for r in signal['suffix'].values() if r['status']=='candidate']
        first19=parse_descriptors(bits[f'frame_{frame+1:02d}_h1_retained_bits'],19)
        first20=header['first_layout20']
        quantities_scaled=first19['entries'][0]['quantity']==2*first20['entries'][0]['quantity']
        if not quantities_scaled:raise RuntimeError('First-entry scale equivalence not observed')
        row={'legacy_frame_index':frame,'signal':signal,'primary_signal_contrast':contrast,
             'second_header_end':start+114*blocks,'layout19':h2['layout19'],'layout20':h2['layout20'],
             'prediction_separate':prediction,'prediction_joint':joint,
             'primary_measured_minus_predicted':boundary-end if boundary is not None else None,
             'threshold_candidate_count':len(sensitivity),
             'threshold_boundary_range':max(sensitivity)-min(sensitivity) if sensitivity else None,
             'conditional_at_prediction':{'before_mean':float(before.mean()) if len(before) else None,
                                          'after_mean':float(after.mean()) if len(after) else None,
                                          'before_windows':len(before),'after_windows':len(after),
                                          'note':'Uses the field prediction; not an independent boundary search.'},
             'h1_19bit_count_is_twice_20bit_count':quantities_scaled,
             'unknown_19bit_units':sorted({e['mcs_candidate'] for e in h2['layout19']['entries']} - set(UNITS)),
             'primary_window_permutation_controls':nulls[frame]}
        rows.append(row)
        print('comparison frame',frame,'predicted',end,'measured',boundary,'contrast',contrast,
              'error',row['primary_measured_minus_predicted'],'tail',prediction['remaining_resources'],flush=True)
    contrast_rows=[r for r in rows if r['primary_signal_contrast']]
    errors=[r['primary_measured_minus_predicted'] for r in contrast_rows]
    result={'schema':'starlink-allocation-observability/v1','seed':seed,'input_sha256':source_hash,
            'header_evidence_sha256':digest(header_path),'header_bit_evidence_sha256':digest(bit_path),
            'signal_only_calculated_before_headers':True,'profile_window_resources':32,'profile_discarded_resources':16,
            'primary_threshold':.45,'thresholds':THRESHOLDS,'frozen_units':UNITS,'frame_count':len(rows),
            'frames':rows,'signal_contrast_cases':len(contrast_rows),
            'contrast_rmse_resources':float(np.sqrt(np.mean(np.array(errors,dtype=float)**2))) if errors else None,
            'contrast_max_abs_error_resources':max(map(abs,errors)) if errors else None,
            'raw_pn_omitted_candidates':sum(r['signal']['raw_suffix']['status']=='candidate' for r in rows),
            'wrong_pn_candidates':sum(r['signal']['wrong_pn_suffix']['status']=='candidate' for r in rows),
            'raw_pn_omitted_contrast_candidates':sum(upward_contrast(r['signal']['raw_suffix']) for r in rows),
            'wrong_pn_contrast_candidates':sum(upward_contrast(r['signal']['wrong_pn_suffix']) for r in rows),
            'permutation_trials':len(rows)*199,
            'permutation_candidates':sum(n['status']=='candidate' for r in rows for n in r['primary_window_permutation_controls']),
            'permutation_contrast_candidates':sum(upward_contrast(n) for r in rows for n in r['primary_window_permutation_controls']),
            'layout19_fitted_unit_diagnostic':fit_unit_diagnostic(rows,19),
            'layout20_fitted_unit_diagnostic':fit_unit_diagnostic(rows,20),
            'limitations':['Same capture and inherited timing/PN/units; no independent acquisition or end-to-end RF receiver.',
                           'Reference-like payload may be indistinguishable from padding in this statistic.',
                           'Signal contrast selection uses only the signal, not agreement with header predictions.',
                           'Short suffixes and fixed block alignment constrain primary detector sensitivity; missing estimates are retained.',
                           'Window permutation preserves local coherence/histogram, not an iid RF-noise null.',
                           'LS one-change model is misspecified for multi-region data; apparent changes are diagnostic candidates.',
                           '19-bit unknown units and unit/count rescaling prevent unique layout identification.',
                           'Endpoint comparisons alone do not certify codeword rounding, parity, application content or spacecraft identity.']}
    (out/'allocation_observability.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(out/'coherence_profiles.npz',**profiles)
    with (out/'allocation_endpoints.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['legacy_frame','predicted_separate','predicted_joint','remaining','measured_primary','contrast','error','threshold_range','after_prediction_windows'])
        for r in rows:writer.writerow([r['legacy_frame_index'],r['prediction_separate']['end_resource'],r['prediction_joint']['end_resource'],
                                      r['prediction_separate']['remaining_resources'],r['signal']['suffix']['0.45']['boundary_resource'],
                                      r['primary_signal_contrast'],r['primary_measured_minus_predicted'],r['threshold_boundary_range'],
                                      r['conditional_at_prediction']['after_windows']])
    print('signal contrast cases',len(contrast_rows),'RMSE',result['contrast_rmse_resources'],'max error',result['contrast_max_abs_error_resources'],flush=True)


if __name__=='__main__':main()
