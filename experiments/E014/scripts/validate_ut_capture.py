"""Blind-to-CRC acquisition on a second RF capture; frozen P/H only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import sys
import numpy as np
from scipy.io import loadmat

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context,legacy_artifact
from ut_receiver_core import read_ci16,resample_480,repetition_candidates,refine_pss,capture_grid,pilot_ratios,pilot_residual,decode_variable_pdu
from code5_core import LOADED,equalize_grid,reference_gain
from layout_core import NONPILOT
from receiver_models import pn_reference
from educational_fec_demo import word_bits,encode_block
from sparse_code5_core import syndrome,normalized_min_sum


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def compact(header):return {k:v for k,v in header.items() if k not in ['bits','reencoded']}


def main():
    require_run_context();out=experiment_output('E014');out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin';source_hash=digest(source)
    if source_hash!='c69e82022d45173cdcaf1a5fdc8a2b864e2d61e85ae819af54b16002044adae3':raise RuntimeError('New capture changed')
    generator_path=ROOT/'runs/v010-code5-parity-02/artifacts/experiments/E012/results/code5_observations.npz'
    check_path=ROOT/'runs/v011-sparse-code5/artifacts/experiments/E013/results/sparse_code5_arrays.npz'
    p=np.load(generator_path,allow_pickle=False)['parity_generator'];h=np.load(check_path,allow_pickle=False)['check_matrix']
    p_hash=hashlib.sha256(p.tobytes()).hexdigest();h_hash=hashlib.sha256(h.tobytes()).hexdigest()
    if p_hash!='d82157accb735e9c96c5fb3181136ea10969f97e51ee5fb516a4f78f5ec4f01c' or h_hash!='cf50cc7d89fe6c5a18b6393182065f5114344a7311c405d6df9b1fc5cae6dfe5':raise RuntimeError('Frozen matrices changed')
    cache=ROOT/'.cache/ut250_c69e820_480_6145.npy';meta_path=cache.with_suffix('.json')
    if cache.exists():
        metadata=json.loads(meta_path.read_text(encoding='utf8'))
        if metadata['sha256']!=digest(cache):raise RuntimeError('Resampling cache corrupted')
    else:
        cache.parent.mkdir(parents=True,exist_ok=True);print('Resampling unmodified raw IQ',flush=True)
        values=resample_480(read_ci16(source));np.save(cache,values)
        metadata={'source_sha256':source_hash,'sha256':digest(cache),'samples':len(values),'rate_hz':480000000,'up':48,'down':25,'taps':6145}
        meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf8');del values
    values=np.load(cache,mmap_mode='r',allow_pickle=False);candidates=repetition_candidates(values)
    print('Repetition candidates',len(candidates),flush=True)
    sss_path=legacy_artifact('reference/qin/sssVec.mat');sss=loadmat(sss_path)['sssVecFull'].ravel()[LOADED%1024]
    prefix=word_bits(0x3048)[16:];known_signs=1-2*encode_block(np.r_[prefix,np.zeros(16,dtype=np.uint8)])[:48].astype(float)
    cases=[];word_rows=[];arrays={};symbols=[];hard_words=[];decoded_words=[]
    for index,candidate in enumerate(candidates):
        case={'candidate_index':index,**candidate};cases.append(case)
        try:
            match=refine_pss(values,candidate['rough_epoch'],candidate['coarse_frequency_hz']);case['pss']=match
            if match['score']<.3:case['status']='pss_template_score_rejected';continue
            epoch=match['epoch'];scale=1.;frequency=match['cfo_hz'];history=[]
            for iteration in range(2):
                spectra,slots=capture_grid(values,epoch,scale,frequency)
                try:residual=pilot_residual(pilot_ratios(spectra,slots,sss),slots[1:])
                except ValueError as error:
                    history.append({'iteration':iteration,'status':str(error)});break
                history.append({'iteration':iteration,**residual});frequency+=residual['frequency_hz'];scale*=1+residual['clock_fraction']
            spectra,slots=capture_grid(values,epoch,scale,frequency)
            equal=equalize_grid(spectra[1:],spectra[0],sss,slots[1:]);flat=equal['nonpilot'].ravel()
            gain=reference_gain(flat[:48],known_signs);normalized=flat/gain
            case.update(receiver_refinement=history,cfo_hz=frequency,sampling_scale=scale,complete_frame=bool(slots[-1]==301),last_observed_slot=int(slots[-1]),
                        reserved_pilot_evm=equal['reserved_pilot_evm'],reserved_pilot_errors=equal['reserved_pilot_label_errors'],reserved_pilot_count=equal['reserved_pilot_count'],
                        known_prefix_gain_real=float(gain.real),known_prefix_gain_imag=float(gain.imag))
            first=decode_variable_pdu(normalized[:228],'h1');case['h1']=compact(first)
            if first['crc_ok'] is not True:case['status']='h1_not_validated';continue
            arrays[f'candidate_{index}_h1_bits']=first['bits'];arrays[f'candidate_{index}_h1_symbols']=normalized[:228]
            entry=first['layout20']['entries'][0]
            if entry['mcs_candidate']!=5 or entry['quantity']<1:case['status']='unsupported_first_allocation';continue
            count=entry['quantity'];second_start=228+2048*count;case['predicted_h2_start']=second_start
            if second_start+114<=len(normalized):
                second=decode_variable_pdu(normalized[second_start:second_start+798],'h2');case['h2']=compact(second)
                if 'bits' in second:arrays[f'candidate_{index}_h2_bits']=second['bits']
            for word in range(count):
                start=228+word*2048;stop=start+2048
                if stop>len(normalized):break
                observed=normalized[start:stop];bits=(observed.real<0).astype(np.uint8)
                predicted=(bits[:1024]@p)%2;pre_failed=int(syndrome(bits[None,:],h).sum())
                decoded=normalized_min_sum(4*observed.real/.02,h);decoded_bits=decoded.pop('bits')
                reencoded=np.r_[decoded_bits[:1024],(decoded_bits[:1024]@p)%2]
                wrong={}
                for shift in [-1,1]:
                    b=(normalized[start+shift:stop+shift].real<0).astype(np.uint8)
                    if len(b)==2048:wrong[str(shift)]=int(np.sum(((b[:1024]@p)%2)!=b[1024:]))
                positions=np.arange(start,stop);ks=NONPILOT[positions%1004];ss=2+positions//1004
                rotor=np.array([pn_reference(int(k),np.array([int(s)]))[0] for s,k in zip(ss,ks)])
                nopn=(observed*rotor).real<0
                row={'candidate_index':index,'word':word,'start_resource':start,'complete_frame':case['complete_frame'],
                     'initial_failed_checks':pre_failed,'initial_parity_disagreements':int(np.sum(predicted!=bits[1024:])),
                     'decoder':decoded,'changed_hard_bits':int(np.sum(decoded_bits!=bits)),
                     'decoded_reencoding_disagreements':int(np.sum(reencoded!=decoded_bits)),
                     'signal_nmse':float(np.sum(abs(observed-(1-2*reencoded.astype(float)))**2)/np.sum(abs(observed)**2)),
                     'offset_parity_disagreements':wrong,'no_pn_parity_disagreements':int(np.sum(((nopn[:1024].astype(np.uint8)@p)%2)!=nopn[1024:]))}
                word_rows.append(row);symbols.append(observed);hard_words.append(bits);decoded_words.append(decoded_bits)
            case['status']='h1_and_first_body_processed'
            print('candidate',index,'epoch',round(epoch,2),'complete',case['complete_frame'],'CFO',round(frequency,2),'ppm',round((scale-1)*1e6,3),
                  'pilot EVM',round(equal['reserved_pilot_evm'],3),'H1',first['words_hex'],'H2',case.get('h2',{}).get('words_hex'),flush=True)
        except (ValueError,IndexError) as error:case['status']='receiver_or_bounds_failure';case['reason']=str(error)
    valid=[c for c in cases if c.get('h1',{}).get('crc_ok') is True]
    result={'schema':'starlink-ut-transfer/v1','source_sha256':source_hash,'resampling':metadata,'generator_sha256':p_hash,'check_matrix_sha256':h_hash,
            'sss_reference_sha256':digest(sss_path),'candidate_status_counts':dict(Counter(c['status'] for c in cases)),
            'candidates':cases,'words':word_rows,'h1_crc_count':len(valid),'h2_crc_count':sum(c.get('h2',{}).get('crc_ok') is True for c in cases),
            'complete_validated_h1_frames':sum(c['complete_frame'] for c in valid),'code5_words':len(word_rows),
            'complete_frame_code5_words':sum(w['complete_frame'] for w in word_rows),
            'code5_initial_exact_parity_words':sum(w['initial_parity_disagreements']==0 for w in word_rows),
            'code5_decoder_converged':sum(w['decoder']['converged'] for w in word_rows),
            'code5_decoder_changed_bits':sum(w['changed_hard_bits'] for w in word_rows),'matrices_refitted':False,
            'limitations':['Published dataset and high-level results inspected; new RF capture relative to old data, not blind-to-literature validation.',
                           'Front-end acquisition/tracking use known sync/pilots, not CRC or codeword parity optimization. All detected candidates and failures retained.',
                           'Resource completeness and full-frame completeness reported separately; labels not inferred as on-air counters or spacecraft IDs.',
                           'Code membership and signal consistency do not authenticate source bytes or measure transmitted BER. User narrowband messages unchanged.']}
    arrays.update(observed_code5_symbols=np.array(symbols),hard_code5_words=np.array(hard_words),decoded_code5_words=np.array(decoded_words))
    (out/'ut_transfer.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(out/'ut_observations.npz',**arrays)
    print('Transfer summary:',{k:result[k] for k in ['candidate_status_counts','h1_crc_count','h2_crc_count','complete_validated_h1_frames','code5_words','code5_initial_exact_parity_words','code5_decoder_converged']},flush=True)


if __name__=='__main__':main()
