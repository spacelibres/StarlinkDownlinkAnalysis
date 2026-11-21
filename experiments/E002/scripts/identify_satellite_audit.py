"""Trace retained IQ to archived orbital nominations; do not call them decoded IDs."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E002')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import collections,csv,datetime,hashlib,json

ROOT=EXTERNAL_LEO_ROOT
R=ROOT/'reports';OUT=experiment_output('E002', '')
SOURCE=R/'2026_10_18_dsx01_pilot_clock'
curves=json.loads((R/'2026_10_17_robust_curve_reassociation/partition/DSX01-S05/curves.json').read_text())
tle=json.loads((R/'2026_10_18_dsx01_retention_localization/causal_tles.json').read_text())['DSX01-S05']
records=[];tracks=set()
for group in ['DSX01-S05','paired_RX']:
    meta=json.loads((SOURCE/group/'selection.json').read_text())
    for w in meta['windows']:
        matched=[c for c in curves if w['observation_id'] in c['observation_ids']]
        assert len(matched)==1
        track=matched[0]['key'];tracks.add(track)
        records.append({'source_group':group,'array_key':'iq_'+str(w['number']),'observation_id':w['observation_id'],'track':track,'sample_rate_hz':meta['sample_rate_hz'],'duration_s':.02,'context_time_s':w['context_time_s'],'receiver':w['receiver'],'actual_rf_hz':w['probe']['actual_rf_hz']})
variants=collections.defaultdict(list);sources={}
for family in ['2026_10_18_dsx01_receiver_geometry/DSX01-S05','2026_10_18_dsx01_pilot_clock/localization/DSX01-S05']:
    for p in sorted((R/family).glob('*/endpoints.json')):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();doc=json.loads(p.read_text())
        fits=[(measure+'/'+name,fit) for measure,v in doc.items() for name,fit in v['fits'].items()] if 'localization' in family else list(doc.items())
        for name,fit in fits:
            for row in fit.get('rows',[]):
                if row.get('component') in tracks:
                    variants[row['component']].append({'source':str(p),'model':p.parent.name,'variant':name,'fit_latitude_deg':fit.get('latitude_deg'),'fit_longitude_deg':fit.get('longitude_deg'),**{k:row.get(k) for k in ['candidate_norad','compatible','feasible_candidates','rms_hz','bias_hz']}})
summaries={}
for track in sorted(tracks):
    v=variants[track];candidates=[]
    for n,count in collections.Counter(a['candidate_norad'] for a in v).items():
        sat=tle['satellites'][str(n)];assert int(sat['lines'][0][2:7])==n
        candidates.append({'norad':n,'name':sat['provenance']['name'],'archived_variants_nominating':count,'compatible_variants':sum(a['compatible'] is True for a in v if a['candidate_norad']==n),'rms_hz_min':min(a['rms_hz'] for a in v if a['candidate_norad']==n),'rms_hz_max':max(a['rms_hz'] for a in v if a['candidate_norad']==n),'tle_lines':sat['lines'],'tle_provenance':sat['provenance']})
    summaries[track]={'candidates':candidates,'variant_count':len(v),'status':'unconfirmed_orbital_nomination','warning':'Variants share data and may inherit frozen nominations. Their counts are not independent votes or probabilities.'}
for row in records:row['candidates']=summaries[row['track']]['candidates']
result={'schema':'local-satellite-identity-audit/v1','decoded_identity':False,'npz':{'sample_origin_utc':datetime.datetime.fromtimestamp(tle['origin_utc_ns']/1e9,datetime.timezone.utc).isoformat(),'windows':records,'track_summaries':summaries,'model_details':dict(variants)},'wav':{'file':'USER_IQ.wav','status':'unidentified','recording_location_known':False,'recording_timezone_known':False,'filename_date_order_known':False,'missing_metadata_confirmed_by_user':True,'reason':'Captured center-band signal has no validated ID-field decoder; absent acquisition geometry/time prevents reliable TLE association. Git upload date is not acquisition time.'},'source_sha256':sources,'method_boundary':'This audits existing orbital hypotheses; it does not decode a spacecraft ID from pilot symbols or establish authenticated emitter identity.'}
(artifact_path('satellite_identity_audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
with (artifact_path('satellite_identity_candidates.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['source_group','array_key','track','context_time_s','receiver','actual_rf_hz','candidate_norad','candidate_name','status']);writer.writeheader()
    for row in records:
        for c in row['candidates']:writer.writerow({**{k:row[k] for k in ['source_group','array_key','track','context_time_s','receiver','actual_rf_hz']},'candidate_norad':c['norad'],'candidate_name':c['name'],'status':'UNCONFIRMED_ORBITAL_NOMINATION'})
for track,s in summaries.items():print(track,[(c['norad'],c['name'],c['archived_variants_nominating'],c['compatible_variants']) for c in s['candidates']])
print('Verified',len(records),'array-to-track bindings and',len(summaries),'track summaries; WAV remains unidentified.')
