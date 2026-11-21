"""Recover both PDU headers using resource bounds and self-described block counts."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import re
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import experiment_output,require_run_context
from protocol_core import extract_header,decode_header,FRAME_START,FRAME_PERIOD
from layout_core import eligible_first_headers,extract_resources,second_header_start,decode_second_at,decode_second_values,parse_descriptors

def compact(decoded):
    return {k:v for k,v in decoded.items() if k not in ['retained_bits','soft','reencoded']}

def main():
    require_run_context();parser=argparse.ArgumentParser();parser.add_argument('--controls',type=int,default=512)
    args=parser.parse_args();out=experiment_output('E010');out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy'
    with source.open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    if source_hash!='a218fc9d307b7c0f3acb8263529ab5a690618119604adbcc4847ea87e6272abf':raise RuntimeError('Input capture changed')
    iq=np.load(source,mmap_mode='r',allow_pickle=False);frames=eligible_first_headers(len(iq));rows=[];arrays={};raw=[];misplaced=[]
    for frame in frames:
        y,k=extract_header(iq,frame);first=decode_header(y,k);first['words_hex']=[f'0x{x:08X}' for x in first['words']]
        if not first['crc_ok']:raise RuntimeError('First header failed; refusing location search')
        start=second_header_start(first['retained_bits']);second=decode_second_at(iq,frame,start)
        for name,decoded in [('h1',first),('h2',second)]:
            for key in ['retained_bits','soft','reencoded']:
                if key in decoded:arrays[f'frame_{frame+1:02d}_{name}_{key}']=decoded[key]
        record={'legacy_frame_index':frame,'capture_ordinal':frame+1,
                'pss_start_sample':FRAME_START+frame*FRAME_PERIOD,
                'pss_start_observed':FRAME_START+frame*FRAME_PERIOD>=0,
                'first_header':compact(first),'second_header':compact(second),
                'first_layout20':parse_descriptors(first['retained_bits'],20)}
        rows.append(record)
        full,_,_,ks=extract_resources(iq,frame,start,456);raw.append((full,ks))
        for offset in [-114,-1,1,114]:
            wrong=decode_second_at(iq,frame,start+offset)
            misplaced.append({'legacy_frame_index':frame,'offset_resources':offset,
                              'precrc_admitted':wrong['precrc_admitted'],'crc_ok':wrong['crc_ok'],
                              'coded_hard_disagreements':wrong.get('coded_hard_disagreements')})
        print('frame',frame,'H1',first['crc_ok'],'H2',second['crc_ok'],'start',start,
              'N',second['blocks'],'D',second['entry_count'],second.get('words_hex'),flush=True)
    seed=2026112010;rng=np.random.default_rng(seed);nulls=[]
    for index in range(args.controls):
        row=index%len(raw);values,k=raw[row]
        control=decode_second_values(values[rng.permutation(len(values))],k)
        nulls.append({'source_legacy_frame':frames[row],'precrc_admitted':control['precrc_admitted'],
                      'needs_more_resources':control.get('needs_more_resources',False),'crc_ok':control['crc_ok'],
                      'coded_hard_disagreements':control.get('coded_hard_disagreements')})
        if (index+1)%128==0:print('permutation controls',index+1,'/',args.controls,flush=True)
    # Only now compare decoded output with the old, CRC-searched archival result.
    archive=ROOT/'references/legacy_research/second_header/analysis/starangle_out/ku_header2_positions.txt'
    archived={}
    for line in archive.read_text(encoding='utf8').splitlines():
        match=re.match(r'\s*(\d+)\s+0x[0-9A-Fa-f]+\s+\(\s*\d+,\s*\d+\)\s+\d+\s+[\d.]+\s+(.*?)\s+CRC OK',line)
        if match:archived[int(match[1])]=[int(x,16) for x in re.findall(r'0x[0-9A-Fa-f]+',match[2])]
    comparisons=[{'frame':r['legacy_frame_index'],'matches':r['second_header'].get('words')==archived.get(r['legacy_frame_index'])} for r in rows if r['legacy_frame_index']>=0]
    count1=sum(r['first_header']['crc_ok'] for r in rows);count2=sum(r['second_header']['crc_ok'] is True for r in rows)
    joint2=sum(r['second_header']['crc_ok'] is True and r['second_header'].get('coded_hard_disagreements')==0 for r in rows)
    report={'schema':'starlink-full-header-layout/v1','input_npy_sha256':source_hash,'source_kind':'Same historical public wideband capture, not user narrowband RF',
            'seed':seed,'frames':rows,'first_headers_crc':count1,'second_headers_crc':count2,'frame_count':len(frames),
            'second_headers_crc_and_exact_hard_reencoding':joint2,
            'predicted_second_slots':dict(Counter(str(r['second_header']['start_slot']) for r in rows)),
            'decoded_block_counts':dict(Counter(str(r['second_header']['blocks']) for r in rows)),
            'archive_comparison':comparisons,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'wrong_positions':misplaced,'resource_permutation_controls':nulls,
            'wrong_position_crc_passed':sum(r['crc_ok'] is True for r in misplaced),
            'permutation_precrc_admitted':sum(r['precrc_admitted'] for r in nulls),
            'permutation_crc_passed':sum(r['crc_ok'] is True for r in nulls),
            'crc_used_for_location_blockcount_or_polarity':False,
            'user_narrowband_validated_messages':0,
            'limitations':['Historical waveform/encoder parameters were learned on the same capture; this is inherited-model replay, not independent discovery.',
                           'H1 polarity uses known first16 bits; H2 uses known first8 bits. CRC is never a branch selector.',
                           'Prefix block-count/entry-count and the 20-bit descriptors are conditional models; header checks do not prove field-name uniqueness.',
                           'A newly included capture-edge header is additional recovered content from the same recording, not a new RF observation.',
                           'Offsets/permutations are finite controls. Some wrong-position candidates span slots; no universal false-positive claim.',
                           'Forced zero trellis tails are model constraints, not independent evidence. Application payload and spacecraft identity remain unresolved.']}
    (out/'full_header_layout.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(out/'full_header_observations.npz',**arrays)
    with (out/'header_descriptors.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['legacy_frame','capture_ordinal','header','entry','mcs_candidate','quantity','layout_width','crc_ok'])
        for row in rows:
            for label,layout,crc in [('h1',row['first_layout20'],row['first_header']['crc_ok']),('h2',row['second_header'].get('layout20',{}),row['second_header']['crc_ok'])]:
                for index,entry in enumerate(layout.get('entries',[])):
                    writer.writerow([row['legacy_frame_index'],row['capture_ordinal'],label,index,entry['mcs_candidate'],entry['quantity'],20,crc])
    print('CRC H1/H2:',count1,count2,'of',len(frames),'exact H2:',joint2,'archive matches:',sum(r['matches'] for r in comparisons),'/',len(comparisons),flush=True)
    if not(count1==count2==joint2==len(frames)
           and all(r['first_header']['coded_hard_disagreements']==0 for r in rows)
           and len(comparisons)==len(archived)==17 and all(r['matches'] for r in comparisons)):
        raise SystemExit('Two-header gate not achieved; evidence retained')

if __name__=='__main__':main()
