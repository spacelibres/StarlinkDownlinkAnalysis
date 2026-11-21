"""Explicit compatibility paths for the imported historical experiment scripts.

New experiments should use explicit input/output arguments. This resolver keeps
the legacy artifact names traceable through the versioned migration manifest.
"""
from functools import lru_cache
import json
import os
from pathlib import Path,PurePosixPath
import sys

PROJECT_ROOT=Path(__file__).resolve().parent
RAW_WAV=PROJECT_ROOT/'data/raw/wav/USER_IQ.wav'
LEO_SNAPSHOT=PROJECT_ROOT/'data/raw/leo_tracker_snapshot'
EXTERNAL_LEO_ROOT=Path(os.environ.get('STARLINK_EXTERNAL_LEO_ROOT','../leo-tracker-reduxredux-main'))
VI_ROOT=PROJECT_ROOT/'references/visual_identity'

@lru_cache(maxsize=1)
def layout():
    return json.loads((PROJECT_ROOT/'provenance/layout_manifest.json').read_text(encoding='utf8'))

def _redirect(path):
    run=os.environ.get('STARLINK_RUN_DIR')
    if not run:return path
    relative=path.relative_to(PROJECT_ROOT)
    belongs=relative.parts[0] in ['reports','references','learning']
    belongs=belongs or (len(relative.parts)>2 and relative.parts[0]=='experiments' and relative.parts[2] in ['results','figures'])
    belongs=belongs or relative.parts[:2]==('data','derived')
    return Path(run)/relative if belongs else path

def experiment_output(experiment,prefix=''):
    if prefix=='learning':path=PROJECT_ROOT/'learning'
    elif prefix=='npz_iq_exports':path=PROJECT_ROOT/'data/derived/ci16'
    else:path=PROJECT_ROOT/'experiments'/experiment/'results'
    return _redirect(path)

def legacy_artifact(relative,prefix='',experiment='E007'):
    relative=str(relative).replace('\\','/')
    logical=(PurePosixPath(prefix)/relative).as_posix() if prefix else relative
    candidate=PurePosixPath(logical)
    if candidate.is_absolute() or '..' in candidate.parts:raise ValueError('Artifact path must remain inside the project')
    key='analysis/'+logical
    mapped=layout()['legacy_paths'].get(key) or layout()['legacy_paths'].get(logical)
    if mapped is not None:return _redirect(PROJECT_ROOT/mapped)
    directories={'reference':'references','learning':'learning','npz_iq_exports':'data/derived/ci16','frame_candidates':'data/derived/frame_candidates'}
    for old,new in directories.items():
        if logical==old or logical.startswith(old+'/'):
            return _redirect(PROJECT_ROOT/new/PurePosixPath(logical).relative_to(old))
    return experiment_output(experiment,prefix)/relative

def require_run_context():
    if not os.environ.get('STARLINK_RUN_DIR'):
        raise SystemExit('Use python tools/run_experiment.py <script-stem> so archived results are not overwritten.')

# Historical local imports remain explicit, with unique script basenames.
for directory in sorted((PROJECT_ROOT/'experiments').glob('E*/scripts')):
    sys.path.insert(0,str(directory))
