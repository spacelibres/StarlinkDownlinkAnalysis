"""One-time, byte-exact read-only snapshots of legacy payload labels/calibration."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import shutil

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
LEGACY=Path(r'LOCAL_USER_DIRECTORY\Documents\starlink_beacon_iq-master\starlink_beacon_iq-master')


def main():
    manifest=ROOT/'provenance/legacy_payload_import.json'
    if manifest.exists():raise SystemExit('Refusing to overwrite source evidence')
    selected=['analysis/68_ku_payload_equalised.py','analysis/probes/p3_pilot_slope.py',
              'docs/reports/53_ku_payload_first_step.md','analysis/53_ku_channel_profile.py']
    files=[]
    for relative in selected:
        source=LEGACY/relative;destination=ROOT/'references/legacy_research/payload'/relative
        if destination.exists():raise FileExistsError(destination)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        expected=hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:raise RuntimeError('Copy differs')
        files.append({'source':str(source),'snapshot':destination.relative_to(ROOT).as_posix(),
                      'sha256':expected,'bytes':source.stat().st_size})
    manifest.write_text(json.dumps({'schema':'starlink-legacy-import/v1','utc':datetime.now(timezone.utc).isoformat(),
                                   'files':files,'scope':'Legacy labels and calibration ideas only; mislabeled H2 intervals are not adopted as code5 payload.'},indent=2),encoding='utf8')
    print('Read-only source snapshots:',len(files))


if __name__=='__main__':main()
