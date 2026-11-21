"""Build an auditable campaign index and numerical differences from run records."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context

def differences(a,b,path='',rows=None):
    if rows is None:rows=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for key in a.keys()|b.keys():
            if key not in a or key not in b:rows.append({'path':path+'/'+key,'kind':'key difference'})
            else:differences(a[key],b[key],path+'/'+key,rows)
    elif isinstance(a,list) and isinstance(b,list):
        if len(a)!=len(b):rows.append({'path':path,'kind':'length difference'})
        else:
            for i,(x,y) in enumerate(zip(a,b)):differences(x,y,path+'/'+str(i),rows)
    elif a!=b:
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and not isinstance(a,bool):
            rows.append({'path':path,'kind':'numeric','absolute_delta':abs(a-b),'run':a,'historical':b})
        else:rows.append({'path':path,'kind':'value difference','run':a,'historical':b})
    return rows

def checked(relative):
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Invalid input record: '+str(relative))
    return path

def main():
    require_run_context();parser=argparse.ArgumentParser()
    for name in ['campaign','npz-audit','wav-audit','code-audit']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    paths={key:checked(value) for key,value in vars(args).items()}
    loaded={key:json.loads(path.read_text(encoding='utf8')) for key,path in paths.items()}
    jobs=[]
    for job in sorted(loaded['campaign']['jobs'],key=lambda r:r['id']):
        run=ROOT/'runs'/job['run_id'];manifest=json.loads((run/'manifest.json').read_text(encoding='utf8'))
        row={**job,'canonical_mutations':manifest['canonical_evidence_mutations'],'comparisons':[]}
        for relative in manifest['changed_artifacts']:
            historical=ROOT/relative;actual=run/'artifacts'/relative
            delta={'artifact':relative}
            if not historical.exists():delta['kind']='new artifact'
            elif actual.suffix=='.json':
                values=differences(json.loads(actual.read_text(encoding='utf8')),json.loads(historical.read_text(encoding='utf8')))
                delta['differences']=values
            elif actual.suffix=='.npz':
                with np.load(actual) as a,np.load(historical) as b:
                    delta['array_differences']={}
                    for key in a.files:
                        if key not in b:delta['array_differences'][key]={'kind':'new key'}
                        elif a[key].shape!=b[key].shape:delta['array_differences'][key]={'kind':'shape difference','run':list(a[key].shape),'historical':list(b[key].shape)}
                        elif not np.array_equal(a[key],b[key]):
                            value=float(np.max(abs(a[key]-b[key]))) if a[key].dtype.kind in 'fci' else None
                            delta['array_differences'][key]={'max_absolute_delta':value}
            row['comparisons'].append(delta)
        if job['script']=='measure_burst_edge_alignment':row['comparison_note']='20 ms/400 bootstrap sensitivity vs historical 40 ms/200. A separate exact-parameter run is byte-identical.'
        jobs.append(row)
    data=loaded['npz_audit'];duplicate={(r['dataset'],r['array']) for r in data['receiver_windows'] if r['duplicate_of']}
    counts={}
    for shared in [False,True]:
        rows=[r for r in data['structure_tests'] if r['reference']=='analytic_pn' and r['shared_rank1_pattern_removed']==shared and (r['dataset'],r['array']) not in duplicate]
        counts['shared_rank1_removed_'+str(shared)]={'unique_cases':len(rows),'conditional_family_p_below_001':sum(r['family_max_phase_control_p']<.01 for r in rows)}
    summary={'schema':'starlink-campaign-summary/v1','input_records':{k:p.relative_to(ROOT).as_posix() for k,p in paths.items()},
             'baseline_jobs':jobs,'all_jobs_exit_zero':all(j['returncode']==0 for j in jobs),
             'canonical_mutations':sum(len(j['canonical_mutations']) for j in jobs),
             'npz_unique_windows':data['unique_sample_windows'],'npz_counts':counts,
             'wav_wrong_sss_rank_fraction':loaded['wav_audit']['exact_wrong_sss_rank_fraction'],
             'first_header_free_dimensions_even_with_crc_and_known_prefix':loaded['code_audit']['wav_header1_center_capture']['free_binary_dimensions_even_with_crc_and_known_prefix'],
             'validated_unknown_bits':0,'validated_messages':0}
    (out/'campaign_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in summary.items() if k!='baseline_jobs'},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
