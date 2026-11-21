"""Read-only audit/import of explicitly selected legacy evidence, never RF."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-root',type=Path,default=Path(r'LOCAL_USER_DIRECTORY\Documents\starlink_beacon_iq-master\starlink_beacon_iq-master'))
    parser.add_argument('--tracker-root',type=Path,default=Path('../leo-tracker-reduxredux-main'))
    args=parser.parse_args();legacy=args.legacy_root.resolve();tracker=args.tracker_root.resolve()
    manifest=ROOT/'provenance/legacy_research_import.json'
    if manifest.exists():raise SystemExit('Import already recorded; refusing to replace evidence')
    selected=['PROJECT_MAP.md','docs/notes/status_and_gaps.md','docs/notes/proven_facts.md',
              'docs/notes/literature_crosscheck.md','docs/papers/README.md','paper/references.bib','paper/refs_status.md',
              'analysis/kulib.py','analysis/52_ku_header_decode.py','analysis/54_ku_cp_aligned_decode.py','analysis/55_ku_header_alt_estimator.py',
              'docs/reports/48_ku_pn_and_header_decode.md','docs/reports/49_ku_frame_layout_and_headers.md',
              'docs/reports/62_ku_grid_test.md','docs/reports/76_edge_only_vs_full_iq.md',
              'data/downlink_analysis/metadata.json','data/downlink_analysis/README.md','data/downlink_analysis/SHA256SUMS',
              'analysis/starangle_out/ku_header_alt.npz','analysis/starangle_out/ku_header_alt.txt',
              'analysis/starangle_out/ku_header2_final.csv','analysis/starangle_out/ku_pn_full.npz']
    files=[]
    def capture(source,destination):
        source=source.resolve();destination=destination.resolve()
        if not destination.is_relative_to(ROOT):raise ValueError('Destination escapes current project')
        if destination.exists():raise FileExistsError(destination)
        expected=sha(source);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
        if sha(destination)!=expected:raise RuntimeError('Copy verification failed')
        files.append({'source':str(source),'snapshot':destination.relative_to(ROOT).as_posix(),'sha256':expected,'bytes':source.stat().st_size})
    for relative in selected:capture(legacy/relative,ROOT/'references/legacy_research/old_project'/relative)
    for relative in ['AGENTS.md','docs/concepts/starlink-transmissions.md','docs/research/evidence-ledger.md','docs/pipelines/research-analysis.md']:
        capture(tracker/relative,ROOT/'references/legacy_research/leo_tracker'/relative)
    csv=legacy/'data/downlink_analysis/Starlink_Downlink_IQ.csv'
    csv_hash=sha(csv)
    if csv_hash!='797e13e5788f1a54269cc77a8232e259b61933e4bef4d89cd08854c80649f115':raise RuntimeError('Legacy CSV source hash mismatch')
    source=legacy/'data/downlink_analysis/Starlink_Downlink_IQ.npy';array=np.load(source,mmap_mode='r',allow_pickle=False)
    count=0;maximum=0.
    with csv.open('r',encoding='utf8') as stream:
        columns=stream.readline().strip()
        while count<len(array):
            block=np.loadtxt(stream,delimiter=',',max_rows=min(1000000,len(array)-count),ndmin=2)
            values=(block[:,0]+1j*block[:,1]).astype(array.dtype)
            delta=float(np.max(abs(values-array[count:count+len(values)])));maximum=max(maximum,delta)
            if delta:raise RuntimeError('NPY conversion differs from original CSV')
            count+=len(block)
        if stream.read().strip():raise RuntimeError('Unexpected additional CSV rows')
    capture(source,ROOT/'data/raw/legacy_wideband/Starlink_Downlink_IQ.npy')
    papers=[]
    from pypdf import PdfReader
    for path in sorted((legacy/'docs/papers').glob('*.pdf')):
        pdf=PdfReader(path);metadata=pdf.metadata or {}
        papers.append({'source':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'pages':len(pdf.pages),
                       'metadata_title':str(metadata.get('/Title','')),'first_page_text':(pdf.pages[0].extract_text() or '')[:1600],
                       'scope':'Local bibliographic/source audit only; full-paper conclusions are not certified by metadata extraction.'})
    packages=[]
    for directory in sorted((legacy/'third_party').iterdir()):
        if not directory.is_dir():continue
        readmes=[]
        for depth in range(3):
            readmes.extend(directory.glob('/'.join(['*']*depth+['README.md'])))
        paths=sorted(set(readmes),key=lambda p:(len(p.parts),str(p)))[:3]
        packages.append({'directory':str(directory),'readme_files':[{'path':str(p),'sha256':sha(p)} for p in paths],
                         'status':'Located inherited code; not a validated runtime dependency or universal decoder.'})
    simulator=legacy/'third_party/starlink_simulator'
    for path in [simulator/'LICENSE',simulator/'README.md',*sorted((simulator/'utils').glob('*.m'))]:
        capture(path,ROOT/'references/legacy_research/starlink_simulator'/path.relative_to(simulator))
    report={'schema':'starlink-legacy-import/v1','recorded_utc':datetime.now(timezone.utc).isoformat(),
            'files':files,'legacy_root':str(legacy),'tracker_root':str(tracker),
            'csv_verification':{'source':str(csv),'sha256':csv_hash,'columns':columns,'sample_count':count,'complex_dtype':str(array.dtype),'max_absolute_difference':maximum},
            'papers':papers,'code_packages':packages,
            'licensing':'Imported user legacy materials retain their source status; no new license is asserted. The simulator includes its original MIT notice. No remote publication.',
            'scope':'Read-only local source recovery; copied NPY is complex64 derived from the hash-verified published CSV; no new RF or independent capture.'}
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'snapshot_files':len(files),'pdfs_indexed':len(papers),'code_packages_located':len(packages),'csv_samples_matched':count,'csv_npy_max_difference':maximum},indent=2))

if __name__=='__main__':main()
