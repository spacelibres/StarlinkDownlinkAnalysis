"""Losslessly export the retained NPZ IQ windows; never alter source files."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='npz_iq_exports', experiment='E002')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import hashlib,json
import numpy as np

SOURCE=LEO_SNAPSHOT / 'reports/2026_10_18_dsx01_pilot_clock'
OUT=experiment_output('E002', 'npz_iq_exports')
OUT.mkdir(exist_ok=True)
records=[]
def write(name,value,rate,meta,expected=None):
    assert value.ndim==2
    assert np.isfinite(value).all()
    assert np.array_equal(value.real,np.rint(value.real)) and np.array_equal(value.imag,np.rint(value.imag))
    assert max(abs(value.real).max(),abs(value.imag).max())<=32767
    raw=np.stack([value.real,value.imag],axis=-1).astype('<i2')
    path=artifact_path(name);path.write_bytes(raw.tobytes())
    check=np.fromfile(path,dtype='<i2').reshape(raw.shape)
    assert np.array_equal(check[...,0]+1j*check[...,1],value)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected:assert 'sha256:'+digest==expected
    records.append({'file':name,'sample_rate_hz':rate,'sample_count':len(value),'receiver_count':value.shape[1],'duration_s':len(value)/rate,'dtype':'<i2','layout':['sample','receiver','component'],'component_order':['I','Q'],'bytes':path.stat().st_size,'sha256':digest,'source_window_sha256_verified':bool(expected),**meta})

s=json.loads((SOURCE/'DSX01-S05/selection.json').read_text())
with np.load(SOURCE/'DSX01-S05/selected_iq.npz',allow_pickle=False) as z:
    for window in s['windows']:
        key='iq_'+str(window['number']);v=z[key][:,None]
        write('S05_'+key+'_single_rx.ci16',v,s['sample_rate_hz'],{'source_npz':str(SOURCE/'DSX01-S05/selected_iq.npz'),'array_key':key,'receiver_ids':[window['probe']['receiver_id']],'context_time_s':window['context_time_s'],'actual_rf_hz':window['probe']['actual_rf_hz']})
s=json.loads((SOURCE/'paired_RX/selection.json').read_text())
with np.load(SOURCE/'paired_RX/selected_iq.npz',allow_pickle=False) as z:
    for j in (0,2):
        left,right=s['windows'][j:j+2]
        keys=['iq_'+str(w['number']) for w in (left,right)]
        value=np.stack([z[key] for key in keys],axis=1)
        write('paired_window_'+str(j//2)+'_dual_rx.ci16',value,s['sample_rate_hz'],{'source_npz':str(SOURCE/'paired_RX/selected_iq.npz'),'array_keys':keys,'receiver_ids':[w['probe']['receiver_id'] for w in (left,right)],'context_time_s':[w['context_time_s'] for w in (left,right)],'actual_rf_hz':left['probe']['actual_rf_hz']},expected=left['iq_sha256'])
(artifact_path('manifest.json')).write_text(json.dumps({'description':'Lossless CI16 export of retained original IQ snippets, not new acquisition or a continuous recording. Same physical windows occur in both source sets.','files':records},indent=2),encoding='utf8')
print('Verified',len(records),'exports;',sum(r['bytes'] for r in records),'bytes; both dual-RX source window SHA256 values match.')
