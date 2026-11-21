"""Run imported experiments against copied evidence; publish only changed outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def git(*args):
    result=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True)
    return result.stdout.strip() if result.returncode==0 else None

def scripts():
    paths=sorted((ROOT/'experiments').glob('E*/scripts/*.py'))
    paths.append(ROOT/'learning/build_communication_formula_library.py')
    return {p.stem:p for p in paths}

def seed_directories():
    paths=[ROOT/'reports',ROOT/'references',ROOT/'learning',ROOT/'data/derived']
    paths.extend(sorted((ROOT/'experiments').glob('E*/results')))
    paths.extend(sorted((ROOT/'experiments').glob('E*/figures')))
    return paths

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script',nargs='?')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--run-id')
    options,forward=parser.parse_known_args()
    available=scripts()
    if options.list:
        for name,path in available.items():print(name,':',path.relative_to(ROOT).as_posix())
        return 0
    if options.script not in available:parser.error('Select a script from --list')
    if forward[:1]==['--']:forward=forward[1:]
    run_id=options.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+options.script
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,120}',run_id):parser.error('Invalid run id')
    record=ROOT/'runs'/run_id
    workspace=ROOT/'.cache/runs'/run_id/'workspace'
    for path in (record,workspace):
        if not path.resolve().is_relative_to(ROOT):parser.error('Run path escapes project')
        if path.exists():parser.error('Run id already exists; select a new id')
    record.mkdir(parents=True);workspace.mkdir(parents=True)
    before={}
    for directory in seed_directories():
        for source in directory.rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:continue
            relative=source.relative_to(ROOT)
            target=workspace/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)  # Copies, never hard links to immutable evidence.
            before[relative.as_posix()]=digest(source)
    script=available[options.script]
    experiment=script.relative_to(ROOT).parts[1] if script.is_relative_to(ROOT/'experiments') else 'LEARNING'
    environment=os.environ.copy()
    environment.update(STARLINK_RUN_DIR=str(workspace),STARLINK_EXPERIMENT_ID=experiment,PYTHONUTF8='1')
    command=[sys.executable,str(script),*forward]
    code_paths=[ROOT/'project_paths.py',*available.values(),*sorted((ROOT/'tools').glob('*.py'))]
    inputs={p.relative_to(ROOT).as_posix():digest(p) for p in (ROOT/'data/raw').rglob('*') if p.is_file()}
    manifest={'schema':'starlink-run/v1','run_id':run_id,'experiment':experiment,
              'project_version':(ROOT/'VERSION').read_text().strip(),'git_head':git('rev-parse','HEAD'),
              'git_worktree_dirty':bool(git('status','--porcelain')),'command':command,
              'python':sys.version,'working_directory':str(ROOT),
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'raw_input_sha256':inputs,'available_evidence_snapshot_sha256':before,
              'code_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in code_paths},
              'snapshot_note':'Available inputs, not an asserted trace of files actually consumed. Seeds remain encoded in the scripts/results.',
              'external_dependency':'E002 inventory/orbit audits may still require STARLINK_EXTERNAL_LEO_ROOT; external contents are not snapshotted in this run.'}
    (record/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Running',options.script,'; log:',record/'run.log',flush=True)
    with (record/'run.log').open('w',encoding='utf8') as log:
        process=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
    changed={}
    for output in workspace.rglob('*'):
        if not output.is_file() or '__pycache__' in output.parts:continue
        relative=output.relative_to(workspace);sha=digest(output)
        if before.get(relative.as_posix())==sha:continue
        destination=record/'artifacts'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(output,destination)
        changed[relative.as_posix()]={'sha256':sha,'bytes':output.stat().st_size}
    mutated=[key for key,sha in before.items() if not (ROOT/key).is_file() or digest(ROOT/key)!=sha]
    manifest.update(ended_utc=datetime.now(timezone.utc).isoformat(),returncode=process.returncode,
                    changed_artifacts=changed,canonical_evidence_mutations=mutated)
    (record/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Exit:',process.returncode,'; changed artifacts:',len(changed),'; canonical mutations:',len(mutated),flush=True)
    if mutated:print('ERROR: canonical evidence changed:',mutated);return 2
    return process.returncode

if __name__=='__main__':raise SystemExit(main())
