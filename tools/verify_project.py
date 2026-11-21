"""Verify classified evidence and portable active code without rewriting results."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    layout=json.loads((ROOT/'provenance/layout_manifest.json').read_text(encoding='utf8'))
    baseline=json.loads((ROOT/'provenance/baseline_manifest.json').read_text(encoding='utf8'))
    assert len(layout['files'])==baseline['file_count']==183
    assert {r['path'] for r in baseline['files']}=={r['original_path'] for r in layout['files']}
    preserved=0
    for row in layout['files']:
        path=ROOT/row['current_path']
        assert path.resolve().is_relative_to(ROOT) and path.is_file(),path
        if row['preserve_bytes']:
            assert sha(path)==row['sha256_before'],f'Evidence changed: {path}'
            if '--staged' in sys.argv:
                blob=subprocess.run(['git','show',':'+row['current_path']],cwd=ROOT,capture_output=True,check=True).stdout
                if blob.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                    expected=('oid sha256:'+row['sha256_before']).encode()
                    assert expected in blob.splitlines(),f'LFS staged oid changed: {path}'
                else:
                    assert hashlib.sha256(blob).hexdigest()==row['sha256_before'],f'Staged evidence changed: {path}'
            preserved+=1
    snapshots=0
    imports=['leo_input_snapshot','visual_assets','ut_capture_import']+[p.stem for p in sorted((ROOT/'provenance').glob('legacy_*_import.json'))]
    for name in imports:
        if not (ROOT/f'provenance/{name}.json').exists():continue
        for row in json.loads((ROOT/f'provenance/{name}.json').read_text(encoding='utf8'))['files']:
            if 'snapshot' not in row:continue
            assert sha(ROOT/row['snapshot'])==row['sha256'],row['snapshot']
            snapshots+=1
    code=[ROOT/'project_paths.py',*sorted((ROOT/'experiments').glob('E*/scripts/*.py')),
          *sorted((ROOT/'learning').glob('*.py')),*sorted((ROOT/'tools').glob('*.py')),
          *sorted((ROOT/'tests').glob('*.py'))]
    for path in code:
        source=path.read_text(encoding='utf8');ast.parse(source,filename=str(path))
        if path.is_relative_to(ROOT/'experiments') or path.is_relative_to(ROOT/'learning'):
            assert 'analysis' not in source,path
    notebook=json.loads((ROOT/'learning/通信公式与证据学习.ipynb').read_text(encoding='utf8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        assert 'analysis' not in ''.join(cell['source'])
    import project_paths
    for key in layout['legacy_paths']:
        assert project_paths.legacy_artifact(key.removeprefix('analysis/')).is_file(),key
    import numpy as np
    from scipy.io import wavfile
    rate,raw=wavfile.read(project_paths.RAW_WAV,mmap=True)
    slices=0
    for path in (ROOT/'data/derived/frame_candidates').glob('*.npz'):
        with np.load(path) as candidate:
            start=int(candidate['source_first_sample']);values=candidate['raw_iq_int16']
            assert int(candidate['sample_rate_hz'])==rate
            assert np.array_equal(values,raw[start:start+len(values)]),path
            slices+=1
    status=json.loads((ROOT/'experiments/E007/results/decode_progress_summary.json').read_text(encoding='utf8'))
    assert status['stages']['message_fec_crc']=='not reached'
    print(json.dumps({'original_files_accounted_for':len(layout['files']),
                      'byte_exact_evidence_files':preserved,'dependency_snapshot_files':snapshots,
                      'active_python_files_parsed':len(code),'notebook':'portable code cells parse',
                      'candidate_slices_exactly_match_raw_wav':slices,
                      'scope':'Integrity/path verification; not new RF decoding or a full rerun.'},indent=2))

if __name__=='__main__':main()
