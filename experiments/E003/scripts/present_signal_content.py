
# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E003')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv,json,xml.etree.ElementTree as ET
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import resvg_py

OUT=experiment_output('E003', '')
content=json.loads((artifact_path('signal_content_extraction.json')).read_text());probe=json.loads((artifact_path('unknown_subcarrier_probe.json')).read_text())
z=np.load(artifact_path('observed_nonpilot_subcarriers.npz'),allow_pickle=False)
rot=loadmat(artifact_path('reference/qin/referenceTemplate.mat'))['referenceTemplateRotations'];symbol=z['symbol_indices'];cl=(symbol-2)%15;cycle=(symbol-2)//15
rows=[]
for item in probe['results']:
    if not item['candidate_periodic_structure']:continue
    key=item['array_key']+'_k'+str(item['subcarrier_index'])
    y=z[key]*np.exp(-.5j*np.pi*rot[item['subcarrier_index'],symbol-1])[None,:]
    aa=[];bb=[];va=[];vb=[]
    for j in range(2,13):
        a=y[:,(cl==j)&np.isin(cycle,probe['training_whole_15symbol_cycles'])]
        b=y[:,(cl==j)&np.isin(cycle,probe['test_whole_15symbol_cycles'])]
        aa.append(a.mean(axis=1));bb.append(b.mean(axis=1));va.append(np.var(a,axis=1,ddof=1)/a.shape[1]/2);vb.append(np.var(b,axis=1,ddof=1)/b.shape[1]/2)
    a=np.array(aa).T;b=np.array(bb).T;sa=np.sqrt(np.array(va).T);sb=np.sqrt(np.array(vb).T)
    theta=.5*np.angle(np.sum(a*a,axis=1));a=(a*np.exp(-1j*theta[:,None])).real;b=(b*np.exp(-1j*theta[:,None])).real
    ok=(abs(a)>3*sa)&(abs(b)>3*sb)&(a*b>0)
    for frame in range(len(a)):
        for j in range(11):
            rows.append({'source':key,'frame_index':frame,'symbol_class_mod15':j+2,'relative_sign_candidate':('+' if a[frame,j]>0 else '-') if ok[frame,j] else '?','screen_pass':bool(ok[frame,j]),'sigma_threshold':3,'frame_tone_global_sign_ambiguity':True,'crc_validated':False,'meaning':'unknown_not_a_message_bit'})
with (artifact_path('nonpilot_relative_sign_candidates.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={'validated_messages':0,'decoded_semantic_fields':{},'recovered_known_pilot_states':content['pilot']['observed_states'],'known_pilot_reference_agreement':content['pilot']['known_reference_agreement'],'frame_rate_hz_model_fit':content['wav_frame_cadence']['fitted_frame_hz'],'nonpilot_repetition_tests':len(probe['results']),'nonpilot_periodic_candidates':sum(x['candidate_periodic_structure'] for x in probe['results']),'sparse_relative_sign_candidates':sum(x['screen_pass'] for x in rows),'relative_signs_have_global_polarity_ambiguity':True,'full_t_code_recovered':False,'frame_header_recovered':False,'crc_or_fec_validated':False}
(artifact_path('signal_content_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')

pal=json.loads(VI_ROOT / 'assets/palette-tokens.json'.read_text())[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':pal['ink'],'axes.labelcolor':pal['ink'],'axes.edgecolor':pal['line'],'xtick.color':pal['muted'],'ytick.color':pal['muted'],'figure.facecolor':pal['surface'],'axes.facecolor':pal['card']})
fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained');fig.get_layout_engine().set(rect=(0,0,1,.9));fig.suptitle('STARLINK  /  OBSERVED SYMBOL STRUCTURE',x=.04,y=.98,ha='left',fontsize=20,fontweight='bold')
d=np.load(artifact_path('wav_frame_structure.npz'));phase=int(d['phase_sample']);offset=np.arange(-90,91);idx=(phase+offset)%3200
ax=axs[0,0]
for name,color,key in [('Calibration blocks',pal['primaryOnLight'],'train_indices'),('Held blocks',pal['planetOnLight'],'test_indices'),('After 13.8 s',pal['muted'],'post_indices')]:
    ax.plot(offset/2.4,d['shifted_profiles'][d[key]].mean(axis=0)[idx],color=color,label=name)
ax.set(xlabel='Aligned frame phase relative to gap (microseconds)',ylabel='Normalized power',title='01 / Repeated power gap near 750 frames/s');ax.legend(fontsize=9);ax.grid(color=pal['line'],alpha=.5)
ax=axs[0,1];cmap=LinearSegmentedColormap.from_list('orbit',[pal['deep'],pal['primary'],pal['accent']]);im=ax.pcolormesh(offset/2.4,d['time_s'],d['shifted_profiles'][:,idx],shading='auto',vmin=.88,vmax=1.08,cmap=cmap,rasterized=True);ax.set(xlabel='Clock-aligned frame phase (microseconds)',ylabel='Recording time (s)',title='02 / Gap weakens with the moving signal');ax.set_ylim(15.9,0);ax.axhline(13.635,color=pal['planetOnLight'],ls='--',lw=1);fig.colorbar(im,ax=ax,label='Normalized power')
pilot=np.load(artifact_path('observed_pilot_states.npz'));agreement=np.where(pilot['supported'],np.where(pilot['observed_states']==pilot['reference_states'],1,2),0)
ax=axs[1,0];im=ax.imshow(agreement.T,aspect='auto',origin='lower',interpolation='nearest',extent=[2,302,527.5,535.5],cmap=ListedColormap([pal['line'],pal['success'],pal['danger']]),vmin=0,vmax=2);ax.set(xlabel='OFDM symbol index',ylabel='Pilot subcarrier index',title='03 / Measured pilot states vs published reference');cb=fig.colorbar(im,ax=ax,ticks=[0,1,2]);cb.ax.set_yticklabels(['Guard / erased','Agreement','Mismatch'])
ax=axs[1,1];xx=np.arange(len(probe['results']));actual=[x['repetition_15_score'] for x in probe['results']];null=[max(x['shifted_reference_control_scores']) for x in probe['results']];ax.bar(xx-.18,actual,width=.36,color=pal['primaryOnLight'],label='Reference removed, period 15');ax.bar(xx+.18,null,width=.36,color=pal['planetOnLight'],label='Largest shifted-reference control');ax.set_xticks(xx,[x['array_key']+'\nk'+str(x['subcarrier_index']) for x in probe['results']],fontsize=8);ax.set(ylabel='Train / test pattern correlation',title='04 / Nonpilot structure candidates; no message decoded');ax.legend(fontsize=8);ax.grid(axis='y',color=pal['line'],alpha=.5)
fig.savefig(artifact_path('signal_content.svg'));plt.close(fig)
tree=ET.parse(artifact_path('signal_content.svg'));svg=tree.getroot();svg.set('width','1536');svg.set('height','960');logo=ET.parse(str(VI_ROOT / 'assets/horizontal-primary.svg')).getroot();logo.set('x','918');logo.set('y','0');logo.set('width','220');logo.set('height','49')
for el in logo.iter():
    if el.get('data-stroke-role')=='planet':el.set('stroke',pal['planetOnLight'])
svg.append(logo);tree.write(artifact_path('signal_content.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('signal_content.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('signal_content.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
print(json.dumps(summary,indent=2),flush=True)
