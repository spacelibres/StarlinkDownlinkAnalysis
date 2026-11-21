"""Recover complete sparse checks before observing words, then test decoding."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context
from code5_core import recover_generator
from sparse_code5_core import find_sparse_checks,independent_checks,binary_rank,normalized_min_sum,syndrome


def summarize(cases):
    return {'trials':len(cases),'converged':sum(x['converged'] for x in cases),
            'exact_reference_words':sum(x['reference_bit_disagreements']==0 for x in cases),
            'valid_but_wrong_words':sum(x['converged'] and x['reference_bit_disagreements']>0 for x in cases),
            'initial_bit_errors':sum(x['input_bit_errors'] for x in cases),
            'residual_bit_errors':sum(x['reference_bit_disagreements'] for x in cases),
            'max_iterations_used':max((x['iterations'] for x in cases),default=0)}


def main():
    require_run_context();out=experiment_output('E013');out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runs/v010-code5-parity-02/artifacts/experiments/E012/results/code5_observations.npz'
    with source.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    original=np.load(source,allow_pickle=False);p=original['parity_generator']
    p_hash=hashlib.sha256(p.tobytes()).hexdigest()
    if p_hash!='d82157accb735e9c96c5fb3181136ea10969f97e51ee5fb516a4f78f5ec4f01c':raise RuntimeError('Frozen generator changed')
    search=find_sparse_checks(p,progress=lambda side,anchor,n:print('search',side,'anchor',anchor,'seeds',n,flush=True))
    selected=independent_checks(search['seeds'],1024,2048,128);h=selected.pop('matrix')
    rank=binary_rank(h);hp=h[:,1024:];hu=h[:,:1024];hp_rank=binary_rank(hp)
    residual=(hu.T+(p@hp.T))%2;orthogonal=int(np.count_nonzero(residual))
    reverse=recover_generator(hp,hu)
    reconstruction=bool(reverse['generator'] is not None and np.array_equal(reverse['generator'].T,p))
    result={'schema':'starlink-sparse-code5/v1','source_npz_sha256':source_hash,'generator_sha256':p_hash,
            'search':search,'selected':selected,'check_rank':rank,'parity_submatrix_rank':hp_rank,
            'generator_orthogonality_nonzero_positions':orthogonal,'reverse_generator_matches':reconstruction,
            'check_row_weights':dict(Counter(map(str,h.sum(axis=1).tolist()))),
            'check_column_weights':dict(Counter(map(str,h.sum(axis=0).tolist()))),
            'check_matrix_sha256':hashlib.sha256(h.tobytes()).hexdigest(),
            'search_reads_received_words':False,'minsum_eta':.8,'minsum_max_iterations':60,
            'limitations':['Frozen same-capture generator and QC128/weight<=6 prior; not a unique transmitter implementation.',
                           'No girth, minimum-distance, capacity or universal BER guarantee is established.',
                           'Recorded words had no parity discrepancies before this stage; zero-iteration passage is not evidence of error correction.',
                           'Injected LLR flips and generated AWGN trials are controlled experiments, not additional RF captures.',
                           'Syndrome zero is code membership, not truth or application authentication. Candidate source semantics remain unknown.']}
    arrays={'check_matrix':h}
    if not(rank==hp_rank==1024 and orthogonal==0 and reconstruction):
        (out/'sparse_code5.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        np.savez_compressed(out/'sparse_code5_arrays.npz',**arrays)
        raise SystemExit('Sparse completeness gate failed; numerical evidence retained')
    print('Complete H:',h.shape,'rank',rank,'Hp rank',hp_rank,'row weights',result['check_row_weights'],flush=True)
    # Only now consume received words; sparse representation selected independently of their syndromes.
    bits=original['hard_bits'];observed=original['observed_symbols'];gain=original['known_prefix_gain_by_word']
    normalized=observed/gain[:,None];llr=4*normalized.real/.02
    initial=syndrome(bits,h);seed=2026112013;rng=np.random.default_rng(seed)
    def run_case(values,reference):
        decoded=normalized_min_sum(values,h);word=decoded.pop('bits')
        reencoded=(word[:1024]@p)%2
        return {**decoded,'input_bit_errors':int(np.count_nonzero((values<0)!=reference)),
                'reference_bit_disagreements':int(np.count_nonzero(word!=reference)),
                'changed_hard_bits':int(np.count_nonzero(word!=(values<0))),
                'reencoding_disagreements':int(np.count_nonzero(reencoded!=word[1024:]))},word
    real=[];decoded_words=[]
    for j in range(len(bits)):
        case,word=run_case(llr[j],bits[j]);case['word_index']=j
        predicted=gain[j]*(1-2*word.astype(float))
        case['original_observation_nmse']=float(np.sum(abs(observed[j]-predicted)**2)/np.sum(abs(observed[j])**2))
        real.append(case);decoded_words.append(word)
    result.update(real_initial_failed_checks=int(initial.sum()),real_cases=real,real_summary=summarize(real),
                  phase_scaled_hard_differences_from_E012=int(np.count_nonzero((llr<0)!=bits)),
                  real_signal_nmse=float(np.sum(abs(observed-gain[:,None]*(1-2*np.array(decoded_words).astype(float)))**2)/np.sum(abs(observed)**2)))
    print('Real received words',result['real_summary'],flush=True)
    injections={};injected_positions={}
    for count in [1,4,16]:
        cases=[];positions=[]
        for j in range(len(bits)):
            where=rng.choice(2048,count,replace=False);values=llr[j].copy();values[where]*=-1
            case,_=run_case(values,bits[j]);case['word_index']=j;cases.append(case);positions.append(where)
        injections[str(count)]={'cases':cases,'summary':summarize(cases)}
        injected_positions[f'injection_{count}_positions']=np.array(positions)
        print('Injected LLR flips',count,injections[str(count)]['summary'],flush=True)
    simulations={}
    for sigma in [.4,.6,.8,1.]:
        cases=[]
        for index in range(16):
            u=rng.integers(0,2,1024,dtype=np.uint8);truth=np.r_[u,(u@p)%2]
            noisy=1-2*truth.astype(float)+rng.normal(0,sigma,2048)
            case,_=run_case(2*noisy/(sigma*sigma),truth);case['trial']=index;cases.append(case)
        simulations[str(sigma)]={'cases':cases,'summary':summarize(cases)}
        print('Synthetic real-AWGN sigma',sigma,simulations[str(sigma)]['summary'],flush=True)
    result.update(seed=seed,injected_llr_flips=injections,synthetic_real_awgn=simulations,
                  real_likelihood_assumed_complex_noise_scale=.02,
                  llr_scaling_note='Unclipped normalized min-sum is homogeneous for global positive LLR scaling; scale repetitions are not independent robustness evidence.')
    arrays.update(decoded_real_words=np.array(decoded_words),**injected_positions)
    (out/'sparse_code5.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(out/'sparse_code5_arrays.npz',**arrays)


if __name__=='__main__':main()
