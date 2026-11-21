"""Append a bounded read-only legacy source snapshot for the second-header study."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
LEGACY=Path(r'LOCAL_USER_DIRECTORY\Documents\starlink_beacon_iq-master\starlink_beacon_iq-master')

def main():
    manifest=ROOT/'provenance/legacy_second_header_import.json'
    if manifest.exists():raise SystemExit('Second-header sources already snapshotted; refusing replacement')
    names=['analysis/57_ku_header2.py','analysis/58_ku_header2_positions.py','analysis/59_ku_header_fields.py',
           'analysis/60_ku_header_semantics.py','analysis/65_ku_header2_judge.py','analysis/66_ku_header2_final.py',
           'analysis/starangle_out/ku_header2_positions.txt','analysis/starangle_out/ku_header_fields.csv',
           'docs/reports/52_ku_header2_semantics.md']
    rows=[]
    for name in names:
        source=(LEGACY/name).resolve();target=(ROOT/'references/legacy_research/second_header'/name).resolve()
        if not source.is_relative_to(LEGACY) or not target.is_relative_to(ROOT):raise ValueError('Path escapes declared roots')
        if target.exists():raise FileExistsError(target)
        expected=hashlib.sha256(source.read_bytes()).hexdigest();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise RuntimeError('Snapshot byte mismatch')
        rows.append({'source':str(source),'snapshot':target.relative_to(ROOT).as_posix(),'bytes':target.stat().st_size,'sha256':expected})
    manifest.write_text(json.dumps({'schema':'starlink-legacy-import/v1','utc':datetime.now(timezone.utc).isoformat(),
                                    'files':rows,'scope':'Historical sources only; their CRC-guided searches are not used as current branch selectors.'},indent=2),encoding='utf8')
    print('Verified immutable source snapshots:',len(rows))

if __name__=='__main__':main()
