"""Can complementary receivers predict the same proposed repeated code positions?"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import LEO_SNAPSHOT,experiment_output,require_run_context
from receiver_models import pn_reference

def remove_shared(training,test):
    training=training-training.mean(axis=1,keepdims=True)
    test=test-test.mean(axis=1,keepdims=True)
    _,_,v=np.linalg.svd(training,full_matrices=False);axis=v[0]
    return (training-(training@axis.conj())[:,None]*axis,
            test-(test@axis.conj())[:,None]*axis)

def overlap_prediction(atr,ate,btr,bte):
    atr,ate=remove_shared(atr,ate);btr,bte=remove_shared(btr,bte)
    candidates=[]
    for offset in [-1,0,1]:
        aa=atr[max(0,-offset):min(len(atr),len(btr)-offset)]
        at=ate[max(0,-offset):min(len(ate),len(bte)-offset)]
        bb=btr[max(0,offset):min(len(btr),len(atr)+offset)]
        bt=bte[max(0,offset):min(len(bte),len(ate)+offset)]
        gain=np.sum(bb.conj()*aa,axis=1)/np.maximum(np.sum(abs(bb)**2,axis=1),1e-30)
        prediction=bb*gain[:,None]
        train=float(np.real(np.vdot(aa,prediction))/max(np.linalg.norm(aa)*np.linalg.norm(prediction),1e-30))
        prediction=bt*gain[:,None]
        score=float(np.real(np.vdot(at,prediction))/max(np.linalg.norm(at)*np.linalg.norm(prediction),1e-30))
        candidates.append({'offset':offset,'training_score':train,'test_score':score,'frame_pairs':len(aa)})
    return max(candidates,key=lambda r:r['training_score'])

def main():
    require_run_context();parser=argparse.ArgumentParser();parser.add_argument('--observations',required=True)
    parser.add_argument('--audit',required=True);args=parser.parse_args()
    paths=[(ROOT/p).resolve() for p in [args.observations,args.audit]]
    if any(not p.is_relative_to(ROOT) or not p.is_file() for p in paths):raise ValueError('Invalid input record')
    audit=json.loads(paths[1].read_text(encoding='utf8'))
    train=np.array(audit['calibration_cycles_after_first_two']);test=np.array(audit['reserved_cycles_after_first_two'])
    slots=np.arange(2,302);seed=2026112010;rng=np.random.default_rng(seed)
    cases=[('iq_0',527,'iq_1',527),('iq_0',537,'iq_1',525),('iq_2',537,'iq_3',525)]
    rows=[];nulls=[]
    with np.load(paths[0]) as data:
        def profile(key,k):
            name='paired_RX_'+key;ids=data[name+'_indices'];col=list(ids).index(k)
            values=data[name+'_unknown'][:,:,col]*pn_reference(k,slots).conj()[None,:]
            values=values.reshape(len(values),20,15)[:,2:]
            return values[:,train].mean(axis=1),values[:,test].mean(axis=1)
        for ak,ac,bk,bc in cases:
            # Bit position mapping is a conditional 60-bit/16-shift hypothesis.
            pa=(ac-16*np.arange(2,17))%60;pb=(bc-16*np.arange(2,17))%60
            allowed_a=set(pa[2:13]);allowed_b=set(pb[2:13]);common=[int(x) for x in sorted(allowed_a&allowed_b)]
            ai=[list(pa).index(bit) for bit in common];bi=[list(pb).index(bit) for bit in common]
            a,at=profile(ak,ac);b,bt=profile(bk,bc);a=a[:,ai];at=at[:,ai];b=b[:,bi];bt=bt[:,bi]
            actual=overlap_prediction(a,at,b,bt);control=[]
            for _ in range(999):
                permutation=rng.permutation(len(common))
                control.append(overlap_prediction(a,at,b[:,permutation],bt[:,permutation])['test_score'])
            nulls.append(control)
            rows.append({'rx_a':ak,'carrier_a':ac,'rx_b':bk,'carrier_b':bc,'common_bit_positions':common,
                         **actual,'null_max':max(control),'validates_joint_codeword':False})
    joint=np.max(nulls,axis=0)
    for row in rows:
        row['family_permutation_p']=(1+int(np.sum(joint>=row['test_score'])))/1000
        row['supports_transfer_under_tested_model']=row['family_permutation_p']<.01
    metadata=json.loads((LEO_SNAPSHOT/'reports/2026_10_18_dsx01_pilot_clock/paired_RX/selection.json').read_text(encoding='utf8'))
    report={'scope':'Cross-receiver held-cycle prediction, per-frame gain nuisance, common rank-one reference removed; no receiver merging.',
            'seed':seed,'permutations_per_case':999,'family_cases':len(rows),'results':rows,
            'inputs_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'receiver_contexts':[{'array':'iq_'+str(w['number']),'context_time_s':w['context_time_s'],'anchor_sample':w['winner']['integer_epoch_sample']+w['winner']['fractional_epoch_offset_samples']} for w in metadata['windows']],
            'validated_unknown_bits':0,'validated_messages':0,
            'limitations':['Exploratory follow-up on already analyzed captures; no untouched RF dataset.',
                           'Same coded-position mapping across receivers is a hypothesis; field identity is not assumed.',
                           'Frame offset selected only on calibration cycles from -1,0,1; controls repeat this search.',
                           'Bit-position permutation assumes residual exchangeability; different class noise can violate it.',
                           'Receiver context times are not proof of sample-clock synchronization or the same transmit frame.',
                           'Positive transfer would not certify global polarity, a complete codeword or CRC.']}
    out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    (out/'cross_receiver_information.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
