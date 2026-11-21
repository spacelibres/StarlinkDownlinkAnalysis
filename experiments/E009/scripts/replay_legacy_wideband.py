"""Recompute legacy first PDU headers from original wideband IQ, with nulls."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context
from protocol_core import extract_header,decode_header,frame_count

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    require_run_context();parser=argparse.ArgumentParser();parser.add_argument('--controls',type=int,default=512)
    args=parser.parse_args();out=experiment_output('E009');out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((ROOT/'provenance/legacy_research_import.json').read_text(encoding='utf8'))
    for row in provenance['files']:
        if digest(ROOT/row['snapshot'])!=row['sha256']:raise RuntimeError('Imported evidence changed: '+row['snapshot'])
    source=ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
    iq=np.load(source,mmap_mode='r',allow_pickle=False);count=frame_count(len(iq));rows=[];arrays={};raw=[]
    seed=2026112009;rng=np.random.default_rng(seed);assignment=rng.permutation(count);calibration=assignment[:4]
    for frame in range(count):
        y,carriers=extract_header(iq,frame);raw.append(y)
        decoded=decode_header(y,carriers)
        for key in ['soft','retained_bits','reencoded']:arrays[f'frame_{frame:02d}_{key}']=decoded.pop(key)
        decoded.update(frame=frame,words_hex=[f'0x{word:08X}' for word in decoded['words']],
                       group='conditional_replay_calibration' if frame in calibration else 'conditional_replay_reserved')
        rows.append(decoded)
        print('frame',frame,decoded['words_hex'],'CRC',decoded['crc_ok'],'hard errors',decoded['coded_hard_disagreements'],flush=True)
    controls=[]
    for index in range(args.controls):
        frame=index%count;values=raw[frame][rng.permutation(len(carriers))]
        decoded=decode_header(values,carriers)
        controls.append({'source_frame':frame,'crc_ok':decoded['crc_ok'],
                         'coded_hard_disagreements':decoded['coded_hard_disagreements'],
                         'prefix_coded_correlation':decoded['prefix_coded_correlation']})
        if (index+1)%128==0:print('resource-order controls',index+1,'/',args.controls,flush=True)
    # Expected archived output is opened only AFTER measured IQ decoding/nulls.
    expected_path=ROOT/'references/legacy_research/old_project/analysis/starangle_out/ku_header_alt.npz'
    with np.load(expected_path,allow_pickle=False) as archived:expected=archived['words']
    matches=np.array_equal(np.array([r['words'] for r in rows]),expected)
    # Explicit full-grid interoperability check, independent of header outputs.
    from receiver_models import pn_reference
    sequence=np.zeros(32767,dtype=np.uint8);sequence[:15]=[(0x0270>>i)&1 for i in range(15)]
    for j in range(len(sequence)-15):sequence[j+15]=sequence[j]^sequence[j+1]
    slots=np.arange(2,302);carriers_all=np.r_[np.arange(-512,-2),np.arange(2,512)]
    ordinal=np.where(carriers_all<0,carriers_all+512,carriers_all+508);indices=1020*slots[:,None]+ordinal[None,:]-16
    old=np.exp(.5j*np.pi*(sequence[indices%32767]+2*sequence[(indices+16383)%32767]))
    current=np.column_stack([pn_reference(k,slots) for k in carriers_all]);error=float(np.max(abs(old+current)))
    passed=sum(r['crc_ok'] for r in rows);null_pass=sum(r['crc_ok'] for r in controls)
    report={'schema':'starlink-wideband-replay/v1','source_kind':'Existing public legacy 24 ms/480 MS/s capture, not the user narrowband WAV or LEO snippets.',
            'input_npy_sha256':digest(source),'sample_count':len(iq),'sample_rate_hz':480000000,
            'seed':seed,'conditional_calibration_frames':calibration.tolist(),'conditional_reserved_frames':assignment[4:].tolist(),
            'frames':rows,'headers_crc_passed':passed,'headers_tested':count,
            'frozen_prior_prefix_hex':'0x3048','crc_used_for_selection':False,
            'archived_words_all_match':matches,'archive_expected_sha256':digest(expected_path),
            'pn_full_grid_points':int(old.size),'legacy_pn_equals_negative_current_max_error':error,
            'resource_order_nulls':controls,'null_crc_passed':null_pass,'null_trials':len(controls),
            'null_crc_fraction':null_pass/max(len(controls),1),
            'measured_crc_and_exact_hard_reencoding':sum(r['crc_ok'] and r['coded_hard_disagreements']==0 for r in rows),
            'null_crc_and_exact_hard_reencoding':sum(r['crc_ok'] and r['coded_hard_disagreements']==0 for r in controls),
            'user_narrowband_validated_unknown_bits':0,'user_narrowband_validated_messages':0,
            'limitations':['The legacy timing, PN and encoder conventions were historically learned/tuned on this capture; this replay is not an untouched RF holdout.',
                           'Per-frame phase/slope are blind BPSK fits to the header. The known first16 prefix fixes polarity; CRC never selects a branch.',
                           'Current random frame groups are conditional software-regression labels; they do not undo historical selection.',
                           'Resource permutation tests one finite null, not every alternative waveform or false-decoding mechanism.',
                           'CRC8 is error detection, not message authentication or spacecraft identity.',
                           'Header fields and arbitrary application payload are distinct. No user narrowband result is upgraded by this external control.']}
    np.savez_compressed(out/'wideband_first_header_observations.npz',**arrays)
    (out/'wideband_first_header_replay.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    with (out/'wideband_first_header_words.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['frame','group','word0','word1','crc_ok','coded_hard_disagreements'])
        for row in rows:writer.writerow([row['frame'],row['group'],*row['words_hex'],row['crc_ok'],row['coded_hard_disagreements']])
    print('Measured CRC:',passed,'/',count,'; archive exact:',matches,'; null CRC:',null_pass,'/',len(controls),'PN max error',error,flush=True)
    if not(matches and passed==count and error<1e-12):raise SystemExit('Receiver replay gate failed; results retained')

if __name__=='__main__':main()
