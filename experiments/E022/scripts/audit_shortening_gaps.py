"""Test fixed filling hypotheses on archived Code 28 and Code 37 observations."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context
from gap_hypotheses_core import phase_fit, trajectory, recurrence_fit, recurrence_predict
from layout_core import NONPILOT
from receiver_models import pn_bits
from validate_code21 import digest
from validate_code37 import constellation_for

DEGREE=16
NULLS=99
SEED=2026112022
SOURCES = {
    28: ('runs/v019-code28-reproduction/artifacts/experiments/E021/results/code28_reproduction',
         '198225ed52afa7ad4053ea7b9b4523551ed7c5e363be183704acc4aa7abb037d',(8960,9216),
         'primary_hard_bits','final_bits','coordinate_mask'),
    37: ('runs/v015-code37-verified/artifacts/experiments/E017/results/code37_observations',
         '1af099bca2bb10609c3feed0fe5ad8c747beb1d75e6712df92e39fa34bf96788',(4736,5120),
         'hard_bits','corrected_bits','core_coordinate_mask'),
}


def agreement(observed, predicted):
    return {'hits':int(np.count_nonzero(observed==predicted)), 'total':observed.size}


def forecast(sequence, indices, model):
    return trajectory(sequence,indices,model['phase'],model['direction'],model['complement'])


def per_word_phase(sequence,bits):
    half = bits.shape[1]//2
    offset = np.arange(bits.shape[1])
    cases=[]
    for word in bits:
        fitted=phase_fit(sequence,offset[:half],word[:half])
        predicted=forecast(sequence,offset[half:],fitted)
        cases.append({**fitted,'holdout':agreement(word[half:],predicted)})
    return cases


def recurrence_check(bits, training, seed, null_count):
    model = recurrence_fit(bits[training],DEGREE)
    target = bits[~training,DEGREE:]
    predicted=recurrence_predict(bits[~training],model,DEGREE)
    result={'model':model,'holdout':agreement(target,predicted)}
    if null_count:
        rng=np.random.default_rng(seed)
        null=[]
        y_fit=bits[training,DEGREE:]
        for _ in range(null_count):
            cal_offset=rng.integers(DEGREE+1,y_fit.shape[1]-DEGREE,size=len(y_fit))
            val_offset=rng.integers(DEGREE+1,target.shape[1]-DEGREE,size=len(target))
            bootfit=np.stack([np.roll(row,int(k)) for row,k in zip(y_fit,cal_offset,strict=True)])
            bootval=np.stack([np.roll(row,int(k)) for row,k in zip(target,val_offset,strict=True)])
            alternative=recurrence_fit(bits[training],DEGREE,target=bootfit)
            null.append(agreement(bootval,recurrence_predict(bits[~training],alternative,DEGREE))['hits'])
        result.update(null_seed=seed,null_counts=null,null_maximum=int(max(null)),
                      null_greater_or_equal=sum(x>=result['holdout']['hits'] for x in null))
    return result


def audit_format(code):
    relative, expected, interval, raw_key, final_key, mask_key=SOURCES[code]
    path=ROOT/(relative+'.npz')
    sha=digest(path)
    if sha!=expected:
        raise RuntimeError('Frozen observation changed')
    sidecar=path.with_name('code37.json') if code==37 else path.with_suffix('.json')
    records=json.loads(sidecar.read_text(encoding='utf-8'))['words']
    first,stop=interval
    with np.load(path,allow_pickle=False) as data:
        observed=data[raw_key][:,first:stop]
        if not np.array_equal(observed,data[final_key][:,first:stop]) or data[mask_key][first:stop].any():
            raise RuntimeError('Unexplained coordinates were changed or placed in the verified core')
        if code==28:
            symbols=data['primary_symbols']
            constellation=constellation_for(2,False,True)
        else:
            symbols=data['observed_symbols_with_neighbors'][:,1:-1]
            constellation=data['constellation']
        mask=data[mask_key]
        assignments=([('published',data['published_training']),('randomized',data['random_training'])]
                     if code==28 else [('randomized',data['training_frames'])])
    # Conditional bit scores from waveform distance; they are not BER.
    d2=abs(symbols[...,None]-constellation)**2
    signs=(1-2*((np.arange(8)[:,None]>>np.array([2,1,0]))&1))
    margin=np.stack([
        d2[..., bit==-1].min(axis=-1)-d2[..., bit==1].min(axis=-1)
        for bit in signs.T],axis=-1).reshape(len(observed),-1)
    gap_margin=margin[:,first:stop]
    gap_means=np.tanh(gap_margin/.04)
    frames=np.array([r['frame'] for r in records])
    ordinals=np.array([r['word'] for r in records])
    starts=np.array([r['start_resource'] for r in records])
    width=stop-first
    offset=np.arange(width)
    resources=starts[:,None]+(first+offset)//3
    carrier=NONPILOT[resources%1004]
    full_ordinal=np.where(carrier<0,carrier+512,carrier+508)
    index_choices={
        'word_reset':np.broadcast_to(offset,observed.shape),
        'gap_only_continuous':ordinals[:,None]*width+offset,
        'word_bits_continuous':ordinals[:,None]*(16384 if code==28 else 8192)+offset,
        'absolute_nonpilot_resource':3*starts[:,None]+first+offset,
        'absolute_loaded_resource':3*(1020*(resources//1004+2)+full_ordinal)+(first+offset)%3,
    }
    sequence=pn_bits()
    rng=np.random.default_rng(SEED+code)
    positive_shared=trajectory(sequence,index_choices['gap_only_continuous'],857,-1,True)
    positive_shared ^= (rng.random(observed.shape)<.05).astype(np.uint8)
    positive_per_word=trajectory(sequence,offset[None,:],rng.integers(0,len(sequence),size=(len(observed),1)),1,False)
    positive_per_word ^= (rng.random(observed.shape)<.05).astype(np.uint8)
    distances=np.count_nonzero(observed[:,None]!=observed[None,:],axis=2)
    upper=distances[np.triu_indices(len(observed),1)]
    result={'code':code,'observation_sha256':sha,'sidecar_sha256':digest(sidecar),
            'source':relative+'.npz','coordinate_range':list(interval),
            'words':len(observed),'distinct_gaps':len(np.unique(observed,axis=0)),
            'ones':int(observed.sum()), 'bits':observed.size,
            'distance_margin':{'gap_abs_quantiles':np.quantile(abs(gap_margin),[.01,.05,.5]).tolist(),
                               'core_abs_quantiles':np.quantile(abs(margin[:,mask]),[.01,.05,.5]).tolist(),
                               'gap_hard_disagreements':int(np.count_nonzero((gap_margin<0)!=observed))},
            'hamming_distance_min_median_max':[int(upper.min()),float(np.median(upper)),int(upper.max())],
            'individual_phase_prediction':per_word_phase(sequence,observed),
            'synthetic_individual_phase_prediction':per_word_phase(sequence,positive_per_word),
            'synthetic_seed':SEED+code,'synthetic_flip_probability':.05,
            'assignments':[]}
    for i,(name,train) in enumerate(assignments):
        counts=observed[train].sum(axis=0)
        template=(2*counts>train.sum()).astype(np.uint8)
        models={}
        for scenario,indices in index_choices.items():
            fit=phase_fit(sequence,indices[train],observed[train])
            predicted=forecast(sequence,indices[~train],fit)
            hits=agreement(observed[~train],predicted)
            models[scenario]={'fit':fit, 'holdout':hits,
                              'expected_match_with_distance_scores':float(np.mean((1+(1-2*predicted.astype(float))*gap_means[~train])/2))}
        example=index_choices['gap_only_continuous']
        fit=phase_fit(sequence,example[train],positive_shared[train])
        result['assignments'].append({
            'name':name,'training_frames':np.unique(frames[train]).tolist(),'reserved_frames':np.unique(frames[~train]).tolist(),
            'constant_template':agreement(observed[~train],template[None,:]),
            'observed_shared_pn15':models,
            'observed_affine_recurrence':recurrence_check(observed,train,SEED+100*code+i,NULLS),
            'synthetic_shared_pn15':{'fit':fit,'holdout':agreement(positive_shared[~train],forecast(sequence,example[~train],fit))},
            'synthetic_affine_recurrence':recurrence_check(positive_shared,train,SEED+code,0),
        })
    return result


def main():
    require_run_context()
    out=experiment_output('E022')
    out.mkdir(parents=True,exist_ok=True)
    results=[audit_format(code) for code in SOURCES]
    result={'schema':'starlink-shortening-gap-hypotheses/v1','feedback_maximum_order':DEGREE,
            'permutation_trials':NULLS,'pn_sequence_period':len(pn_bits()),'formats':results,
            'scope':'One legacy capture. Negative results refer only to the displayed finite tests.'}
    (out/'shortening_gap_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for row in results:
        print(row['code'],'words',row['words'],'unique',row['distinct_gaps'],
              [(g['name'],g['observed_affine_recurrence']['holdout']) for g in row['assignments']],flush=True)


if __name__=='__main__':
    main()
