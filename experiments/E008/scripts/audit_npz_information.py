"""Cross-cycle receiver prediction and reference-dependent structure, both NPZs."""
from pathlib import Path
import json
import hashlib
import argparse
import sys
import numpy as np
from scipy.io import loadmat

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import LEO_SNAPSHOT,legacy_artifact,experiment_output,require_run_context
from receiver_models import pn_reference,cycle_contrast,fit_frame

def main():
    require_run_context();out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--edge-guard-samples',type=float,default=.6)
    args=parser.parse_args()
    import audit_nonpilot_leakage as core
    seed=2026112008;rng=np.random.default_rng(seed)
    # Exclude the first two cycles and edge classes. Freeze before inspecting IQ.
    assignment=rng.permutation(18);train=np.sort(assignment[:9]);test=np.sort(assignment[9:])
    fulltrain=np.zeros(20,dtype=bool);fulltrain[train+2]=True
    training=fulltrain[np.arange(300)//15]
    # First two cycles are neither channel fitting nor held-cycle pattern evidence.
    phases=rng.uniform(-np.pi,np.pi,(999,len(test)))
    rot=loadmat(legacy_artifact('reference/qin/referenceTemplate.mat'))['referenceTemplateRotations']
    slots=np.arange(2,302);rows=[];receiver=[];arrays={};nulls=[]
    base=LEO_SNAPSHOT/'reports/2026_10_18_dsx01_pilot_clock'
    seen_samples={}
    for dataset in ['DSX01-S05','paired_RX']:
        metadata=json.loads((base/dataset/'selection.json').read_text(encoding='utf8'))
        with np.load(base/dataset/'selected_iq.npz') as source:
            for window in metadata['windows']:
                number=window['number'];key='iq_'+str(number);fits=[]
                sample_hash=hashlib.sha256(source[key].tobytes()).hexdigest()
                duplicate_of=seen_samples.get(sample_hash)
                seen_samples.setdefault(sample_hash,dataset+'/'+key)
                for frame in range(16):
                    fit=fit_frame(source[key],window['winner'],frame,core.expected,training,metadata['sample_rate_hz'],args.edge_guard_samples)
                    if fit is None:break
                    fits.append(fit)
                if not fits:raise RuntimeError('No complete frame: '+dataset+'/'+key)
                cube=np.array([f['unknown'] for f in fits]);indices=fits[0]['unknown_indices']
                name=dataset.replace('-','_')+'_'+key
                arrays[name+'_unknown']=cube.astype(np.complex64);arrays[name+'_indices']=indices
                arrays[name+'_cfo_hz']=np.array([f['residual_cfo_hz'] for f in fits])
                errors=np.array([f['heldout_projected_normalized_error'] for f in fits])
                receiver.append({'dataset':dataset,'array':key,'frames':len(fits),'unknown_carriers':indices.tolist(),
                                 'complex_sample_sha256':sample_hash,'duplicate_of':duplicate_of,
                                 'samples_per_symbol':fits[0]['samples_per_symbol'],
                                 'median_reserved_cycle_projected_error':float(np.median(errors)),
                                 'median_pilot_gram_condition':float(np.median([f['pilot_gram_condition'] for f in fits])),
                                 'median_unknown_design_condition':float(np.median([f['unknown_design_condition'] for f in fits])),
                                 'median_unit_noise_variance':float(np.median([f['heldout_unit_noise_variance_by_unknown'] for f in fits])),
                                 'cfo_min_max_hz':[float(arrays[name+'_cfo_hz'].min()),float(arrays[name+'_cfo_hz'].max())],
                                 'cfo_boundary_frames':sum(f['fit_at_search_boundary'] for f in fits),
                                 'warning':'Projected error is a receiver prediction diagnostic, not BER or same-data pilot EVM.'})
                for column,k in enumerate(indices):
                    pn=pn_reference(k,slots);empirical=np.exp(.5j*np.pi*rot[k,slots-1])
                    ratio=pn/empirical
                    for label,reference in [('analytic_pn',pn),('empirical_qin',empirical)]:
                        values=(cube[:,:,column]*reference.conj()[None,:]).reshape(len(fits),20,15)[:,2:]
                        for remove_shared in [False,True]:
                            check=cycle_contrast(values,train,test,phases,remove_shared=remove_shared)
                            nulls.append(check['null_scores'])
                            rows.append({'dataset':dataset,'array':key,'carrier_unsigned':int(k),'reference':label,
                                     'shared_rank1_pattern_removed':remove_shared,
                                     'reserved_cycle_contrast_coherence':check['coherence'],'contrast_score':check['score'],
                                     'reference_ratio_lag15_agreement':float(np.mean(abs(ratio[15:]-ratio[:-15])<1e-10)),
                                     'reference_ratio_is_binary':bool(np.max(abs(ratio.imag))<1e-10),
                                     'validated_unknown_bits':0,'message_crc_passed':False})
                    print(dataset,key,int(k),'PN contrast / frame-varying:',round(rows[-4]['contrast_score'],3),round(rows[-3]['contrast_score'],3),flush=True)
    joint=np.max(nulls,axis=0)
    for row in rows:row['family_max_phase_control_p']=(1+int(np.sum(joint>=row['contrast_score'])))/1000
    # Positive/negative reference-substitution controls do not use measured IQ.
    controls=[]
    for pattern,k in [(p,k) for p in ['constant_pn','unknown_binary_pattern'] for k in [525,526,527,536,537]]:
        random=np.random.default_rng(seed+1)
        gain=np.exp(1j*random.uniform(-np.pi,np.pi,14))
        bits=np.ones((14,15)) if pattern=='constant_pn' else random.choice([-1,1],size=(14,15))
        pn=pn_reference(k,slots);empirical=np.exp(.5j*np.pi*rot[k,slots-1])
        values=gain[:,None]*np.tile(bits,(1,20))*pn[None,:]
        values+=.2*(random.normal(size=values.shape)+1j*random.normal(size=values.shape))
        for label,reference in [('analytic_pn',pn),('empirical_qin',empirical)]:
            check=cycle_contrast((values*reference.conj()[None,:]).reshape(14,20,15)[:,2:],train,test,phases)
            controls.append({'synthetic_pattern':pattern,'carrier_unsigned':k,'reference':label,'score':check['score'],'coherence':check['coherence']})
    control_ok=all(abs(r['score'])<3 for r in controls if r['synthetic_pattern']=='constant_pn' and r['reference']=='analytic_pn') and all(r['score']>3 for r in controls if r['synthetic_pattern']=='unknown_binary_pattern' and r['reference']=='analytic_pn')
    report={'schema':'starlink-information-audit/v1','seed':seed,'phase_controls':999,'edge_guard_samples':args.edge_guard_samples,
            'calibration_cycles_after_first_two':train.tolist(),'reserved_cycles_after_first_two':test.tolist(),
            'classes_used_within_cycle':list(range(2,13)),'receiver_windows':receiver,'structure_tests':rows,
            'synthetic_controls':controls,'synthetic_controls_passed':control_ok,'unique_sample_windows':len(seen_samples),'validated_unknown_bits':0,'validated_messages':0,
            'design_corrections':['First run incorrectly expected reference-artifact structure at carrier 527, whose two references agree over retained classes. Controls now explicitly enumerate all five observed carriers; no IQ-dependent carrier selection.',
                                  'Rank-one shared-pattern removal is a follow-up diagnostic added after initial contrast inspection, not a fresh confirmatory holdout.'],
            'method':'Raw-sample Schur pilot fit with train-cycle-only CFO/channel gains; test centered class contrasts on reserved whole cycles. Constant complex gain per frame is projected out before testing new structure. Shared cycle phase controls, max over all windows/carriers/reference choices.',
            'limitations':['One finite set of captures, not independent spacecraft trials.',
                           'Acquisition anchors inherited from prior known-pilot selection.',
                           'Fit omits detailed receiver filtering and out-of-band leakage.',
                           'Unit-noise variance assumes white complex sample noise and frozen CFO; it is a model sensitivity diagnostic, not measured receiver noise or BER.',
                           'Phase-control inference assumes whole-cycle rotation symmetry under the residual null.',
                           'Cycles are not independent RF captures; no message semantics are assigned to relative structure.',
                           'An empirical-reference periodicity can be explained by the deterministic difference between reference conventions.']}
    np.savez_compressed(out/'npz_crossvalidated_observations.npz',**arrays)
    (out/'npz_information_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    assert control_ok,'Synthetic audit controls failed; evidence retained'
    print('Complete:',len(receiver),'windows;',len(rows),'family-corrected tests; unknown decoded bits: 0',flush=True)

if __name__=='__main__':main()
