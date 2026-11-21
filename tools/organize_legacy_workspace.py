"""One-time, hash-preserving migration from the v0.0.1 workspace layout."""
from pathlib import Path
import ast,hashlib,importlib.metadata,json,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
SCRIPT_GROUPS={
'E001':['analyze_iq','deeper_checks','inspect_waterfall','focus_event'],
'E002':['inventory_npz','export_npz_iq','identify_satellite_audit'],
'E003':['audit_nonpilot_leakage','verify_known_pilots','extract_signal_content','probe_unknown_subcarrier','present_signal_content','projected_nonpilot_decoder','validate_projected_decoder','refine_perframe_content','decode_paired_tcode','check_paired_unknown_overlap','audit_header_observability'],
'E004':['global_signal_audit','check_global_spectral_axis','test_sideband_phase_coupling','test_wideband_drift_link','present_global_hypothesis_tests'],
'E005':['analyze_subband_cyclostationarity','validate_cyclic_estimators','summarize_cyclic_evidence','recover_cyclic_waveform','present_cyclic_evidence'],
'E006':['measure_burst_edge_alignment','wav_coherent_frame_probe','wav_noncoherent_sync_probe','acquire_narrowband_sync_prefix','test_central_gutter_grid'],
'E007':['extract_central_ofdm_candidates','validate_central_narrowband_model','test_central_binary_regions','publish_decode_progress','present_decode_progress']}
OWNERS={name:group for group,names in SCRIPT_GROUPS.items() for name in names}
TITLES={'E001':'基础录波、频谱与事件','E002':'数据清点与来源审计','E003':'导频与局部编码候选','E004':'全局分量和频漂关联','E005':'循环结构与周期波形','E006':'突发沿、同步和中心频率网格','E007':'条件子载波提取与解码进展'}
REPORTS={'分析报告.txt':'E001','NPZ数据说明.txt':'E002','卫星编号识别结果.txt':'E002','信标内容解析阶段结果.txt':'E003','信标内容解析续查结果.txt':'E003','整体信号结构分析.txt':'E004','波形关联假设检验.txt':'E004','分频带循环结构与公式学习结果.txt':'E005','终止沿对齐检验.txt':'E006','分步解码进展.txt':'E007'}

def checked(path):
    path=path.resolve()
    if not path.is_relative_to(ROOT):raise ValueError(f'Outside workspace: {path}')
    return path

def owner(name):
    if name.startswith(('npz_inventory','satellite_')):return 'E002'
    if name.startswith(('global_','wideband_','sideband_')):return 'E004'
    if name.startswith(('subband_','cyclic_','recovered_')):return 'E005'
    if name.startswith(('burst_','wav_coherent_','wav_noncoherent_','wav_pss_','narrowband_','central_gutter_')):return 'E006'
    if name.startswith(('central_','decode_progress','frame_candidate_catalog')):return 'E007'
    if name.startswith(('known_pilot','decoded_known','observed_','nonpilot_','projected_','perframe_','paired_','partial_header','header_observability','unknown_subcarrier','signal_content','wav_frame_')):return 'E003'
    if name.startswith(('metrics','deeper_','spectrum_','overview','waterfall_','event_','signal_event','reference_waterfall','refined_track','candidate_track','drift_scan','transition_','frequency_interpretation')):return 'E001'
    raise ValueError('Unclassified artifact: '+name)

def destination(old):
    path=Path(old)
    if path.suffix=='.wav':return Path('data/raw/wav')/path.name
    relative=path.relative_to('analysis');parts=relative.parts
    if parts[0]=='reference':return Path('references')/Path(*parts[1:])
    if parts[0]=='learning':return Path('learning')/Path(*parts[1:])
    if parts[0]=='npz_iq_exports':return Path('data/derived/ci16')/Path(*parts[1:])
    if parts[0]=='frame_candidates':return Path('data/derived/frame_candidates')/Path(*parts[1:])
    if relative.name=='build_communication_formula_library.py':return Path('learning')/relative.name
    if relative.suffix=='.py':return Path('experiments')/OWNERS[relative.stem]/'scripts'/relative.name
    if relative.name in REPORTS:return Path('reports')/REPORTS[relative.name]/relative.name
    group=owner(relative.name);kind='figures' if relative.suffix in ['.png','.svg'] else 'results'
    return Path('experiments')/group/kind/relative.name

def rewrite_script(path,experiment,prefix=''):
    source=path.read_text(encoding='utf8');tree=ast.parse(source);lines=source.splitlines(keepends=True)
    starts=[0]
    for line in lines:starts.append(starts[-1]+len(line))
    def pos(line,col):return starts[line-1]+len(lines[line-1].encode('utf8')[:col].decode('utf8'))
    def span(node):return pos(node.lineno,node.col_offset),pos(node.end_lineno,node.end_col_offset)
    edits=[]
    def replace(node,value):a,b=span(node);edits.append((a,b,value))
    def visit(node):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['OUT','out'] for t in node.targets) and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Path' for n in ast.walk(node.value)):
            replace(node.value,f'experiment_output({experiment!r}, {prefix!r})');return
        if isinstance(node,ast.Assign) and path.name=='analyze_iq.py' and any(isinstance(t,ast.Name) and t.id=='ROOT' for t in node.targets):
            replace(node.value,'PROJECT_ROOT');return
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='next' and len(node.args)==1:
            argument=node.args[0]
            if isinstance(argument,ast.Call) and isinstance(argument.func,ast.Attribute) and argument.func.attr=='glob' and argument.args and isinstance(argument.args[0],ast.Constant) and argument.args[0].value=='*.wav':
                replace(node,'RAW_WAV');return
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id in ['OUT','out']:
            replace(node,'artifact_path('+ast.get_source_segment(source,node.right)+')');return
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Path' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
            text=node.args[0].value.replace('\\','/')
            external='../leo-tracker-reduxredux-main'
            if text.startswith(external):
                base='EXTERNAL_LEO_ROOT' if path.name in ['inventory_npz.py','identify_satellite_audit.py'] else 'LEO_SNAPSHOT'
                suffix=text[len(external):].lstrip('/');replace(node,base+((' / '+repr(suffix)) if suffix else ''));return
            vi='LOCAL_USER_DIRECTORY/.codex/skills/spacelibre-vi'
            if text.startswith(vi):
                suffix=text[len(vi):].lstrip('/');replace(node,'VI_ROOT'+((' / '+repr(suffix)) if suffix else ''));return
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            vi='LOCAL_USER_DIRECTORY/.codex/skills/spacelibre-vi/'
            if node.value.startswith(vi):replace(node,'str(VI_ROOT / '+repr(node.value[len(vi):])+')');return
        for child in ast.iter_child_nodes(node):visit(child)
    visit(tree)
    for a,b,value in sorted(edits,reverse=True):source=source[:a]+value+source[b:]
    rewritten=ast.parse(source);insert=0
    if rewritten.body and isinstance(rewritten.body[0],ast.Expr) and isinstance(rewritten.body[0].value,ast.Constant) and isinstance(rewritten.body[0].value.value,str):
        insert=sum(len(line) for line in source.splitlines(keepends=True)[:rewritten.body[0].end_lineno])
    prelude=f'''\n# Repository layout adapter; numerical methods below retain their prior equations.\nimport sys as _sys\nfrom pathlib import Path as _Path\nfrom functools import partial as _partial\n_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())\n_sys.path.insert(0, str(_PROJECT_ROOT))\nfrom project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context\nartifact_path = _partial(legacy_artifact, prefix={prefix!r}, experiment={experiment!r})\nif __name__ == "__main__":\n    require_run_context()\n'''
    source=source[:insert]+prelude+source[insert:];ast.parse(source);path.write_text(source,encoding='utf8',newline='\n')

def main():
    baseline=json.loads((ROOT/'provenance/baseline_manifest.json').read_text(encoding='utf8'))
    if (ROOT/'provenance/layout_manifest.json').exists():raise SystemExit('Migration already recorded; refusing to repeat moves.')
    records=[]
    # Resolve and validate EVERY source and destination before moving any file.
    for row in baseline['files']:
        source=checked(ROOT/row['path']);target=checked(ROOT/destination(row['path']))
        if target.exists():raise FileExistsError(target)
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Baseline drift: '+row['path'])
        changed=source.suffix=='.py' and source.parent.name!='reference' and 'reference' not in source.parts and source.name!='educational_fec_demo.py'
        changed=changed or source.suffix=='.ipynb'
        records.append({'original_path':row['path'],'current_path':target.relative_to(ROOT).as_posix(),'sha256_before':row['sha256'],'bytes_before':row['bytes'],'preserve_bytes':not changed})
    (ROOT/'project.json').write_text(json.dumps({'schema':'starlink-scientific-project/v1','name':'StarlinkBeacon','version_file':'VERSION','baseline_version':'0.0.0','experiments':[{'id':key,'title':value,'directory':'experiments/'+key} for key,value in TITLES.items()]},ensure_ascii=False,indent=2),encoding='utf8')
    for record in records:
        source=checked(ROOT/record['original_path']);target=checked(ROOT/record['current_path']);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(target))
    for path in (ROOT/'analysis').rglob('*.pyc'):
        target=checked(ROOT/'.cache/legacy-bytecode'/path.relative_to(ROOT/'analysis'));target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(checked(path)),str(target))
    for key in SCRIPT_GROUPS:
        for kind in ['scripts','results','figures']:(ROOT/'experiments'/key/kind).mkdir(parents=True,exist_ok=True)
    for record in records:
        path=ROOT/record['current_path']
        if path.suffix=='.py' and record['preserve_bytes'] is False:
            experiment=OWNERS.get(path.stem,'LEARNING');prefix='learning' if experiment=='LEARNING' else 'npz_iq_exports' if path.stem=='export_npz_iq' else ''
            rewrite_script(path,experiment,prefix)
    manifest={'schema':'starlink-layout/v1','baseline_tag':'v0.0.1','migration_version':'0.0.2','files':records,'legacy_paths':{r['original_path']:r['current_path'] for r in records}}
    (ROOT/'provenance/layout_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    capture_dependencies()
    print('Migrated',len(records),'baseline files into seven experiment groups.')

def capture_dependencies():
    external=Path('../leo-tracker-reduxredux-main');capsule=[]
    paths=['reports/2026_10_18_dsx01_pilot_clock/'+group+'/'+name for group in ['DSX01-S05','paired_RX'] for name in ['selected_iq.npz','selection.json']]
    paths+=['src/leo/analysis/starlink/templates.py','src/leo/analysis/qam/pilot.py','LICENSE']
    for relative in paths:
        origin=external/relative
        if not origin.is_file():
            capsule.append({'source':str(origin),'status':'not_found','note':'No license or data content is invented.'});continue
        target=checked(ROOT/'data/raw/leo_tracker_snapshot'/relative);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=origin.read_bytes():raise ValueError('Snapshot/source mismatch: '+relative)
        shutil.copyfile(origin,target)
        capsule.append({'source':str(origin),'snapshot':target.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
    (ROOT/'provenance/leo_input_snapshot.json').write_text(json.dumps({'scope':'Read-only source snapshot for retained IQ and numerical constants; not the full external dataset. Missing license is explicitly recorded; no new license is asserted.','files':capsule},indent=2),encoding='utf8')
    vi=Path('LOCAL_USER_DIRECTORY/.codex/skills/spacelibre-vi');visual=[]
    for relative in ['assets/palette-tokens.json','assets/horizontal-primary.svg','references/visual-standard.md']:
        target=checked(ROOT/'references/visual_identity'/relative);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(vi/relative,target)
        visual.append({'source':str(vi/relative),'snapshot':target.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    (ROOT/'provenance/visual_assets.json').write_text(json.dumps({'authoritative_source':'https://github.com/spacelibres/vi','scope':'Exact copies of assets used by the existing plots; no logo geometry changes.','files':visual},indent=2),encoding='utf8')
    dependencies={name:importlib.metadata.version(name) for name in ['numpy','scipy','matplotlib','resvg-py']}
    (ROOT/'requirements.txt').write_text('\n'.join(name+'=='+value for name,value in dependencies.items())+'\n',encoding='utf8')
    (ROOT/'provenance/environment_baseline.json').write_text(json.dumps({'python':sys.version,'dependencies':dependencies},indent=2),encoding='utf8')
    print('Captured input capsule, exact visual assets and tested dependency versions.')

if __name__=='__main__':
    if '--finalize' in sys.argv:capture_dependencies()
    else:main()
