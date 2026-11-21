"""Inspect NPZ schemas without loading large arrays or enabling pickle."""

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
import ast, collections, csv, json, math, struct, zipfile
import numpy as np

ROOT=EXTERNAL_LEO_ROOT
OUT=experiment_output('E002', '')
files=[]; rows=[]; errors=[]
for p in sorted(ROOT.rglob('*.npz')):
    item={'path':str(p),'relative_path':str(p.relative_to(ROOT)),'archive_bytes':p.stat().st_size,'arrays':[]}
    try:
        with zipfile.ZipFile(p) as z:
            for entry in z.infolist():
                if not entry.filename.endswith('.npy'):continue
                with z.open(entry) as stream:
                    magic=stream.read(8)
                    if magic[:6]!=b'\x93NUMPY':raise ValueError('not an NPY array')
                    width=2 if magic[6]==1 else 4
                    size=struct.unpack('<H' if width==2 else '<I',stream.read(width))[0]
                    h=ast.literal_eval(stream.read(size).decode('utf8' if magic[6]==3 else 'latin1'))
                dtype=np.dtype(h['descr']);shape=h['shape']
                a={'name':entry.filename[:-4],'shape':list(shape),'dtype':str(dtype),'kind':dtype.kind,'elements':math.prod(shape),'array_bytes':math.prod(shape)*dtype.itemsize,'compressed_bytes':entry.compress_size}
                item['arrays'].append(a);rows.append({'path':str(p),**a})
        files.append(item)
    except Exception as e:errors.append({'path':str(p),'error':str(e)})
(artifact_path('npz_inventory.json')).write_text(json.dumps({'root':str(ROOT),'files':files,'errors':errors},ensure_ascii=False,indent=2),encoding='utf8')
with (artifact_path('npz_inventory.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['path','name','shape','dtype','kind','elements','array_bytes','compressed_bytes']);writer.writeheader();writer.writerows(rows)
groups=collections.defaultdict(list)
for f in files:groups[tuple(a['name'] for a in f['arrays'])].append(f)
print('FILES',len(files),'ARRAYS',len(rows),'ERRORS',errors,'BYTES',sum(f['archive_bytes'] for f in files),flush=True)
print('COMMON SCHEMAS',flush=True)
for sig,group in sorted(groups.items(),key=lambda x:-len(x[1]))[:22]:
    print('COUNT',len(group),'EXAMPLE',group[0]['relative_path'],'ARRAYS',[(a['name'],a['shape'],a['dtype']) for a in group[0]['arrays'][:14]],flush=True)
print('COMPLEX SCHEMAS',flush=True)
g=collections.Counter((a['name'],tuple(a['shape']),a['dtype']) for a in rows if a['kind']=='c')
for key,count in g.most_common(35):print(count,key,flush=True)
print('LARGEST ARCHIVES',[(f['relative_path'],f['archive_bytes']) for f in sorted(files,key=lambda f:-f['archive_bytes'])[:10]],flush=True)
