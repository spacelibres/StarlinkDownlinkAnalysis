"""Fit code5 only on declared whole words; test reserved parity and controls."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.io import loadmat

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context,legacy_artifact
from code5_core import fft_grid,equalize_grid,LOADED,qc_augment,recover_generator,reference_gain
from allocation_core import predict_end
from educational_fec_demo import encode_block,word_bits


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    require_run_context();out=experiment_output('E012');out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
    source_hash=digest(source)
    if source_hash!='a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf':raise RuntimeError('Capture changed')
    hp=ROOT/'runs/v008-complete-header-layout-02/artifacts/experiments/E010/results/full_header_layout.json'
    bp=hp.with_name('full_header_observations.npz');headers=json.loads(hp.read_text(encoding='utf8'));hb=np.load(bp,allow_pickle=False)
    sp=legacy_artifact('reference/qin/sssVec.mat');sss=loadmat(sp)['sssVecFull'].ravel()[LOADED%1024]
    iq=np.load(source,mmap_mode='r',allow_pickle=False);receiver=[];meta=[];observed=[];uncorrected=[];offsets={-1:[],1:[]};gains=[]
    prefix=word_bits(0x3048)[16:];known_signs=1-2*encode_block(np.r_[prefix,np.zeros(16,dtype=np.uint8)])[:48].astype(float)
    for h in headers['frames']:
        frame=h['legacy_frame_index'];slots=np.arange(2,302);spectra=fft_grid(iq,frame)
        eq=equalize_grid(spectra[1:],spectra[0],sss,slots);flat=eq['nonpilot'].ravel()
        no=fft_grid(iq,frame,correct_cfo=False);eq0=equalize_grid(no[1:],no[0],sss,slots);flat0=eq0['nonpilot'].ravel()
        known=hb[f'frame_{frame+1:02d}_h1_reencoded']
        fixed_gain=reference_gain(flat[:48],known_signs)
        receiver.append({'frame':frame,'reserved_pilot_evm':eq['reserved_pilot_evm'],
                         'reserved_pilot_label_errors':eq['reserved_pilot_label_errors'],'reserved_pilot_count':eq['reserved_pilot_count'],
                         'h1_known_hard_disagreements':int(np.sum((flat[:228].real<0)!=known)),
                         'no_cfo_reserved_pilot_evm':eq0['reserved_pilot_evm'],
                         'no_cfo_h1_hard_disagreements':int(np.sum((flat0[:228].real<0)!=known)),
                         'known_prefix_gain_real':float(fixed_gain.real),'known_prefix_gain_imag':float(fixed_gain.imag)})
        fields=[('first',228,h['first_layout20']['entries'][0]['quantity'])]
        second=h['second_header'];alloc=predict_end(second['start_resource'],second['blocks'],second['layout20']['entries'])
        fields += [(f'later_{j}',item['start_resource'],item['quantity']) for j,item in enumerate(alloc['allocations']) if item['code']==5]
        for label,start,count in fields:
            for word in range(count):
                begin=start+word*2048;stop=begin+2048
                meta.append({'frame':frame,'field':label,'word':word,'start_resource':begin,
                             'training':bool(label=='first' and frame in [-1,1,2])})
                observed.append(flat[begin:stop]);uncorrected.append(flat0[begin:stop])
                gains.append(fixed_gain)
                for shift in offsets:offsets[shift].append((flat[begin+shift:stop+shift].real<0).astype(np.uint8))
        print('frame',frame,'reserved pilot EVM',round(eq['reserved_pilot_evm'],4),'label errors',eq['reserved_pilot_label_errors'],
              'known header disagreements',receiver[-1]['h1_known_hard_disagreements'],flush=True)
    z=np.asarray(observed);z0=np.asarray(uncorrected);bits=(z.real<0).astype(np.uint8);bits0=(z0.real<0).astype(np.uint8)
    train=np.array([r['training'] for r in meta]);first=np.array([r['field']=='first' for r in meta])
    u,v=qc_augment(bits[train,:1024],bits[train,1024:],128);fit=recover_generator(u,v)
    raw_rank=recover_generator(bits[first,:1024],np.zeros((sum(first),1),dtype=np.uint8))['rank']
    training_raw_rank=recover_generator(bits[train,:1024],np.zeros((sum(train),1),dtype=np.uint8))['rank']
    result={'schema':'starlink-code5-parity/v1','input_sha256':source_hash,'header_evidence_sha256':digest(hp),
            'sss_reference_sha256':digest(sp),'known_pilots_per_slot_training':8,'known_pilots_per_slot_reserved':8,
            'receiver':receiver,'words':meta,'first_body_words':int(sum(first)),'all_code5_words':len(meta),
            'training_words':int(sum(train)),'reserved_first_words':int(sum(first&~train)),
            'qc_lift':128,'qc_training_equations':len(u),'raw_first_body_input_rank':raw_rank,'raw_training_input_rank':training_raw_rank,
            'training_fit':{k:v for k,v in fit.items() if k!='generator'},'user_narrowband_messages':0,
            'limitations':['Same inherited capture and published-model parameters. Reserved words are excluded from this generator fit, not previously unseen for the complete research pipeline.',
                           'QC rotations are structural equations, not independent acquisitions. Full input rank only identifies a generator within this model/order.',
                           'No sparse full-rank check matrix is constructed here; a dense generator alone is not proof of an LDPC representation.',
                           'Zero disagreement is against recovered receiver bits, not transmitted truth/BER or spacecraft identity.',
                           'Systematic coordinates may contain source scrambling or higher-layer fields. No authenticated application plaintext is claimed.']}
    arrays={'observed_symbols':z,'observed_symbols_no_cfo':z0,'hard_bits':bits,'hard_bits_no_cfo':bits0,'training_mask':train,'first_body_mask':first,
            'known_prefix_gain_by_word':np.asarray(gains)}
    generator=fit['generator']
    if generator is None:
        result['gate']='training model failed; extraction retained without held-out parameter selection'
        (out/'code5_parity.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        np.savez_compressed(out/'code5_observations.npz',**arrays)
        print('Training fit failed:',result['training_fit'],flush=True)
        raise SystemExit(2)
    prediction=(bits[:,:1024]@generator)%2;errors=np.sum(prediction!=bits[:,1024:],axis=1)
    signs=1-2*prediction.astype(float);parity=z[:,1024:]
    flips=signs.copy();flips[:,::16]*=-1
    scaled_signs=signs*np.asarray(gains)[:,None];scaled_flips=flips*np.asarray(gains)[:,None]
    no_cfo_errors=np.sum(((bits0[:,:1024]@generator)%2)!=bits0[:,1024:],axis=1)
    complement=1-bits;complement_errors=np.sum(((complement[:,:1024]@generator)%2)!=complement[:,1024:],axis=1)
    offset_results={str(shift):np.sum(((np.asarray(b)[:,:1024]@generator)%2)!=np.asarray(b)[:,1024:],axis=1).tolist() for shift,b in offsets.items()}
    seed=2026112012;rng=np.random.default_rng(seed);null=[]
    for _ in range(199):
        shuffled=bits[~train,1024:][:,rng.permutation(1024)]
        null.append(int(np.sum(shuffled!=prediction[~train])))
    for j,row in enumerate(meta):
        row.update(parity_disagreements=int(errors[j]),parity_disagreements_no_cfo=int(no_cfo_errors[j]),
                   complemented_disagreements=int(complement_errors[j]),
                   predicted_parity_nmse=float(np.sum(abs(parity[j]-signs[j])**2)/np.sum(abs(parity[j])**2)),
                   flipped_parity_nmse=float(np.sum(abs(parity[j]-flips[j])**2)/np.sum(abs(parity[j])**2)),
                   known_prefix_scaled_parity_nmse=float(np.sum(abs(parity[j]-scaled_signs[j])**2)/np.sum(abs(parity[j])**2)),
                   known_prefix_scaled_flipped_nmse=float(np.sum(abs(parity[j]-scaled_flips[j])**2)/np.sum(abs(parity[j])**2)))
    result.update(gate='generator recovered; all frozen predictions and failures retained',seed=seed,
                  reserved_first_parity_bits=int(sum(first&~train))*1024,
                  reserved_first_parity_disagreements=int(errors[first&~train].sum()),
                  later_field_parity_disagreements=int(errors[~first].sum()),all_parity_disagreements=int(errors.sum()),
                  all_words_exact=int(np.sum(errors==0)),whole_word_complement_exact=int(np.sum(complement_errors==0)),
                  no_cfo_exact_words=int(np.sum(no_cfo_errors==0)),offset_parity_disagreements=offset_results,
                  parity_permutation_mismatch_counts=null,
                  generator_accepts_all_one_word=bool(np.all(generator.sum(axis=0)%2==1)),
                  generator_commutes_with_qc128=bool(np.array_equal(generator[(np.arange(1024)//128)*128+(np.arange(1024)+1)%128],
                                                      np.roll(generator.reshape(1024,8,128),1,axis=2).reshape(1024,1024))),
                  predicted_reserved_parity_nmse=float(np.sum(abs(parity[~train]-signs[~train])**2)/np.sum(abs(parity[~train])**2)),
                  flipped_reserved_parity_nmse=float(np.sum(abs(parity[~train]-flips[~train])**2)/np.sum(abs(parity[~train])**2)),
                  known_prefix_scaled_reserved_parity_nmse=float(np.sum(abs(parity[~train]-scaled_signs[~train])**2)/np.sum(abs(parity[~train])**2)),
                  known_prefix_scaled_flipped_reserved_nmse=float(np.sum(abs(parity[~train]-scaled_flips[~train])**2)/np.sum(abs(parity[~train])**2)),
                  scaling_note='Follow-up gain from the pre-existing H1 first16 bits (48 coded symbols), never from target parity. Original unit-amplitude metrics retained.')
    arrays.update(parity_generator=generator,predicted_parity=prediction)
    (out/'code5_parity.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(out/'code5_observations.npz',**arrays)
    with (out/'code5_words.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(meta[0]));writer.writeheader();writer.writerows(meta)
    print('fit rank',fit['rank'],'inconsistencies',fit['inconsistent_equations'],'exact words',result['all_words_exact'],'/',len(meta),
          'reserved parity differences',result['reserved_first_parity_disagreements'],flush=True)


if __name__=='__main__':main()
