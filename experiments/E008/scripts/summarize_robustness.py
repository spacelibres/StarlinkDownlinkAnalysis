"""Summarize guard, cross-receiver and event-edge sensitivity without new fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context

def main():
    require_run_context();parser=argparse.ArgumentParser()
    parser.add_argument('--guard-runs',nargs=3,required=True)
    parser.add_argument('--cross-run',required=True);parser.add_argument('--edge-runs',nargs=3,required=True)
    args=parser.parse_args();inputs={}
    def load(run,suffix):
        path=(ROOT/run/'artifacts'/suffix).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError(path)
        inputs[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf8'))
    guards=[]
    for run in args.guard_runs:
        d=load(run,'experiments/E008/results/npz_information_audit.json')
        receivers=[r for r in d['receiver_windows'] if not r['duplicate_of']]
        duplicates={(r['dataset'],r['array']) for r in d['receiver_windows'] if r['duplicate_of']}
        tests=[r for r in d['structure_tests'] if r['reference']=='analytic_pn' and r['shared_rank1_pattern_removed'] and (r['dataset'],r['array']) not in duplicates]
        guards.append({'run':run,'edge_guard_samples':d['edge_guard_samples'],'samples_per_symbol':sorted(set(r['samples_per_symbol'] for r in receivers)),
                       'median_white_unit_noise_variance':float(np.median([r['median_unit_noise_variance'] for r in receivers])),
                       'unique_cases':len(tests),'conditional_p_below_001':sum(r['family_max_phase_control_p']<.01 for r in tests),
                       'passed_cases':[r['dataset']+'/'+r['array']+'/'+str(r['carrier_unsigned']) for r in tests if r['family_max_phase_control_p']<.01]})
    cross=load(args.cross_run,'experiments/E008/results/cross_receiver_information.json')
    edges=[]
    for run in args.edge_runs:
        d=load(run,'experiments/E006/results/burst_edge_alignment_quarter_ms.json')
        edges.append({'run':run,'group_ms':1000*d['noise_group_duration_s'],'replicates':d['bootstrap_replicates'],
                      'pre_groups':len(d['noise_strata_group_indices']['pre']),'post_groups':len(d['noise_strata_group_indices']['post']),
                      'components':[{'component':r['component'],'delay_point_ms':1000*r['delay_vs_negative_lobe_s'],
                                     'conditional_delay_quantiles_95_ms':[1000*x for x in r['conditional_delay_bootstrap_95pct_s']]} for r in d['results']]})
    report={'schema':'starlink-robustness-summary/v1','inputs_sha256':inputs,'guard_sensitivity':guards,
            'unit_noise_variance_ratio_strict_to_default':guards[-1]['median_white_unit_noise_variance']/guards[0]['median_white_unit_noise_variance'],
            'cases_passing_every_guard':sorted(set.intersection(*(set(r['passed_cases']) for r in guards))),
            'cross_receiver_tests':cross['results'],'edge_sensitivity':edges,
            'validated_unknown_bits':0,'validated_messages':0,
            'limits':['Conditional follow-up on the same data; sensitivity settings are not independent confirmations.',
                      'Failure after discarding samples cannot by itself establish a filter artifact: noise variance and condition number increase.',
                      'White-noise model holds CFO fixed and omits colored receiver noise.',
                      'Edge quantiles condition on the stated single-event/resampling model, not guaranteed coverage of physical timing.',
                      'No complete codeword was constructed by merging receivers.']}
    out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    (out/'robustness_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'guards':guards,'noise_variance_ratio':report['unit_noise_variance_ratio_strict_to_default'],'cross_receiver':cross['results']},indent=2))

if __name__=='__main__':main()
