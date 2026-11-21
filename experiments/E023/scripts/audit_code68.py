"""Check Code 68 boundaries and exact compatibility with known codes."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'learning')]
from project_paths import experiment_output,legacy_artifact,require_run_context
from allocation_core import predict_end
from audit_code28_reencoding import fft_binary_parities
from code5_core import LOADED,fft_grid,reference_gain
from code21_ring import binary_blocks,evaluate,pack_blocks,recover
from code28_receiver_core import corrected_grid,crossfit_gain,evaluate_grid
from code68_mapping_core import contract_checks,forced_collisions,nullspace,projection_fit
from educational_fec_demo import encode_block,word_bits
from qc_figure_core import dense_checks,generator_from_shifts
from validate_code21 import SOURCE,HEADERS,digest
from validate_code37 import constellation_for

SOURCE_21='runs/v014-code21-repeat/artifacts/experiments/E016/results/code21_observations.npz'
TABLE='experiments/E018/figure12_shifts.json'
WIDTH=2048
SEED=2026112023


def model_cost(samples,points,scale=.03):
    distances=abs(samples[:,None]-points[None,:])**2
    return -scale*(logsumexp(-distances/scale,axis=1)-np.log(len(points)))


def transition(flat, expected, left, right):
    local=flat[expected-64:expected+64]
    before=model_cost(local,left)
    after=model_cost(local,right)
    score=np.r_[0,np.cumsum(before)]+np.r_[np.cumsum(after[::-1])[::-1],0]
    best=int(np.argmin(score))
    return {'expected':expected,'best_delta':best-64,'contrast_gain':float(min(score[0],score[-1])-score[best]),
            'complete_cost_curve':score.tolist()}


def collect(iq,sss,headers,points):
    z,labels,metadata,diagnostics=[],[],[],[]
    known=word_bits(0x3048)[16:]
    signs=1-2*encode_block(np.r_[known,np.zeros(16,dtype=np.uint8)])[:48].astype(float)
    code28=constellation_for(2,False,True)
    for record in headers:
        f=record['legacy_frame_index']
        h=record['second_header']
        fields=predict_end(h['start_resource'],h['blocks'],h['layout20']['entries'])['allocations']
        target=next((a for a in fields if a['code']==68),None)
        if target is None:
            continue
        _,initial=evaluate_grid(fft_grid(iq,f),sss)
        spectral,stats=evaluate_grid(corrected_grid(iq,f,initial['scale_delta']),sss)
        flat=spectral.ravel()/reference_gain(spectral.ravel()[:48],signs)
        known_field=next(a for a in fields if a['code']==28)
        idx=np.arange(known_field['start_resource'],known_field['end_resource'])
        carrier=idx%1004
        ref=crossfit_gain(flat[idx],idx//1004,carrier,code28)
        nearest=code28[(abs(ref[:,None]-code28)**2).argmin(axis=1)]
        product=ref*nearest.conj()
        gain=(np.bincount(carrier,weights=product.real,minlength=1004)
              +1j*np.bincount(carrier,weights=product.imag,minlength=1004))
        gain/=np.bincount(carrier,weights=abs(nearest)**2,minlength=1004)
        adjusted=(spectral/gain[None,:]).ravel()/reference_gain(spectral.ravel()[:48],signs)
        ii=np.arange(target['start_resource'],target['end_resource'])
        candidate=crossfit_gain(adjusted[ii],ii//1004,ii%1004,points)
        choices=(abs(candidate[:,None]-points)**2).argmin(axis=1)
        tail=adjusted[-512:].mean()
        diagnostics.append({'frame':f,'base_qam_distance':float((abs(flat[ii,None]-points)**2).min(axis=1).mean()),
                            'refined_qam_distance':float((abs(candidate[:,None]-points)**2).min(axis=1).mean()),
                            'qam_point_counts':np.bincount(choices,minlength=16).tolist(),
                            'held_pilot_evm':stats['held_pilot_evm'],'tail_centroid':[float(tail.real),float(tail.imag)],
                            'ending':transition(adjusted,target['end_resource'],points,np.array([tail])),
                            'starting':transition(adjusted,target['start_resource'],code28,points)})
        for j in range(target['quantity']):
            z.append(candidate[j*WIDTH:(j+1)*WIDTH])
            labels.append(choices[j*WIDTH:(j+1)*WIDTH])
            metadata.append({'frame':f,'word':j,'start_resource':int(ii[j*WIDTH])})
    return np.array(z),np.array(labels),metadata,diagnostics


def dense_symbolic(labels,p):
    bit_select=np.zeros((64,len(labels),4),dtype=np.uint8)
    for lane in range(4):
        bit_select[16*lane:16*(lane+1),:,lane]=labels[None,:]==np.arange(16)[:,None]
    flattened=bit_select.reshape(64,-1)
    parity,_=fft_binary_parities(flattened[:,:5120].reshape(64,10,512),p)
    syndrome=parity^flattened[:,5120:]
    packed=np.packbits(np.ascontiguousarray(syndrome.T),axis=1,bitorder='little')
    return packed.view('<u8').ravel()


def summarize(equations):
    basis,rank=nullspace(np.concatenate(equations),64)
    collision=forced_collisions(basis)
    return {'equations':sum(len(v) for v in equations),'rank':rank,'nullity':len(basis),
            'basis_hex':[f'{x:016x}' for x in basis],'forced_collision_count':len(collision),
            'first_forced_collision':list(collision[0]) if collision else None,
            'has_bijection_when_collision_is_forced':False if collision else None}


def qc_fit_bank(bits,training):
    reports=[]
    for lift in [1024,2048]:
        for k in range(1,min(8192//lift,int(training.sum()))):
            split=k*lift
            train_u,train_v=bits[training,:split],bits[training,split:]
            p0,r0,e0=projection_fit(train_u,train_v,lift)
            p,r,e=recover(pack_blocks(train_u,lift),pack_blocks(train_v,lift),lift)
            row={'lift':lift,'systematic_bits':split,'projected_unit_pivots':r0,
                 'projected_inconsistency':e0,'unit_pivots':r,'inconsistent_ring_equations':e}
            if p is not None:
                v=binary_blocks(evaluate(pack_blocks(bits[~training,:split],lift),p,lift),lift)
                row['holdout_differences']=np.count_nonzero(v!=bits[~training,split:],axis=1).tolist()
            reports.append(row)
    return reports


def erasure_check(samples,ids,points,supports,training):
    """Drop all sparse checks incident to the least reliable symbol ranks."""
    distance=abs(samples[:,:,None]-points[None,None,:])**2
    first_two=np.partition(distance,1,axis=-1)[...,:2]
    margins=first_two[...,1]-first_two[...,0]
    order=np.argsort(margins,axis=1,kind='stable')
    equation_rows=np.stack([contract_checks(row,supports) for row in ids])
    symbol_supports=supports//4
    results=[]
    for fraction in [0.,.01,.05,.10,.20]:
        removed=np.zeros(ids.shape,dtype=bool)
        selected=order[:,:int(fraction*ids.shape[1])]
        np.put_along_axis(removed,selected,True,axis=1)
        active=~removed[:,symbol_supports].any(axis=2)
        groups={}
        for name,mask in [('calibration',training),('holdout',~training)]:
            groups[name]=summarize([a[g] for a,g in zip(equation_rows[mask],active[mask],strict=True)])
        results.append({'erasure_fraction':fraction,**groups})
    return results


def main():
    require_run_context()
    out=experiment_output('E023')
    out.mkdir(parents=True,exist_ok=True)
    input_hashes={file:digest(ROOT/file) for file in [SOURCE,HEADERS,SOURCE_21,TABLE]}
    if [input_hashes[x] for x in [SOURCE,HEADERS,SOURCE_21]]!=[
        'a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf',
        'ddf5d4259b0afd01f6f32e00d9466176782915ffcb96fc6ae597006ca6761838',
        'd5b781862d4289aea615621eb95f06a75e10239f8bb100442428974a98bf9ac0']:
        raise RuntimeError('Frozen input changed')
    sss_file=legacy_artifact('reference/qin/sssVec.mat')
    sss=loadmat(sss_file)['sssVecFull'].ravel()[LOADED%1024]
    shifts=np.array(json.loads((ROOT/TABLE).read_text(encoding='utf-8'))['512'])
    supports=np.nonzero(dense_checks(shifts,512))[1].reshape(-1,6)
    with np.load(ROOT/SOURCE_21,allow_pickle=False) as archive:
        p21=archive['parity_generator'][::512].reshape(10,6,512)
    levels=np.array([-3,-1,1,3])/np.sqrt(10)
    points=(levels[:,None]+1j*levels[None,:]).ravel()
    obs,ids,words,receiver=collect(np.load(ROOT/SOURCE,mmap_mode='r'),
                                   sss,json.loads((ROOT/HEADERS).read_text(encoding='utf-8'))['frames'],points)
    training=np.isin([w['frame'] for w in words],[5,9])
    i,q=divmod(15-np.arange(16),4)
    gray=np.array([0,1,3,2],dtype=np.uint8)
    b,c=gray[i],gray[q]
    mapping=np.column_stack((b>>1,c>>1,b&1,c&1))
    bits=mapping[ids].reshape(len(ids),-1)
    variants={}
    for name,eqs in [('code3',[contract_checks(z,supports) for z in ids]),
                     ('code21',[dense_symbolic(z,p21) for z in ids])]:
        mapped=mapping.T.ravel()
        mapping_int=sum(int(x)<<i for i,x in enumerate(mapped))
        variants[name]={'calibration':summarize([v for v,flag in zip(eqs,training,strict=True) if flag]),
                        'holdout':summarize([v for v,flag in zip(eqs,training,strict=True) if not flag]),
                        'failed_checks_per_word_for_gray':[sum((int(a)&mapping_int).bit_count()%2 for a in v) for v in eqs]}
    rng=np.random.default_rng(SEED)
    original=rng.integers(0,2,(len(ids),4096),dtype=np.uint8)
    p3=generator_from_shifts(shifts,512)
    encoded=np.c_[original,binary_blocks(evaluate(pack_blocks(original,512),p3,512),512)]
    matches=(encoded.reshape(len(ids),-1,4)[:,:,None,:]==mapping[None,None,:,:]).all(axis=-1)
    simulated=matches.argmax(axis=-1)
    noisy=simulated.copy()
    disturbed=rng.random(noisy.shape)<.01
    noisy[disturbed]=(noisy[disturbed]+rng.integers(1,16,size=disturbed.sum()))%16
    control={'clean':summarize([contract_checks(y,supports) for y in simulated]),
             'symbol_replacement_probability':.01,
             'changed_symbols':int(disturbed.sum()),'seed':SEED,
             'noisy':summarize([contract_checks(y,supports) for y in noisy])}
    gaussian=[]
    for variance in [.02,.04]:
        samples=points[simulated]+np.sqrt(variance/2)*(rng.normal(size=simulated.shape)+1j*rng.normal(size=simulated.shape))
        quantized=(abs(samples[:,:,None]-points)**2).argmin(axis=2)
        gaussian.append({'complex_variance':variance,'received_label_errors':int(np.count_nonzero(quantized!=simulated)),
                         'erasure_sweep':erasure_check(samples,quantized,points,supports,training)})
    result={'schema':'starlink-code68-audit/v1','inputs_sha256':input_hashes,'sss_sha256':digest(sss_file),
            'receiver':receiver,'word_records':words,'training_frames':[5,9],'reserved_frames':[1,13],
            'reuse_equations':variants,'symbolic_positive_control':control,'unknown_qc_gray':qc_fit_bank(bits,training),
            'code3_observed_erasure_sweep':erasure_check(obs,ids,points,supports,training),
            'code3_gaussian_erasure_controls':gaussian,
            'scope':'Exact equations on tentative sliced labels. Symbol errors can invalidate a correct code. Unknown formats outside each finite bank remain open.'}
    (out/'code68_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'code68_observations.npz',symbols=obs,point_ids=ids,gray_bits=bits,
                        frozen_gray_mapping=mapping,training_words=training)
    print('Words',len(ids),'symbolic ranks',{k:(v['calibration']['rank'],v['holdout']['rank']) for k,v in variants.items()},flush=True)
    print('Unknown QC exact candidates',sum(row['inconsistent_ring_equations']==0 and 'holdout_differences' in row
                                            for row in result['unknown_qc_gray']),flush=True)


if __name__=='__main__':
    main()
