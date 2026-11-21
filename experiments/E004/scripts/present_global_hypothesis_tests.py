"""Scientific summary figure using the approved spacelibre visual assets."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E004')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import resvg_py

OUT=experiment_output('E004', '')
VI=VI_ROOT
pal=json.loads((VI/'assets/palette-tokens.json').read_text(encoding='utf8'))[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':pal['ink'],
    'axes.labelcolor':pal['ink'],'axes.edgecolor':pal['line'],'axes.titleweight':'bold',
    'xtick.color':pal['muted'],'ytick.color':pal['muted'],'figure.facecolor':pal['surface'],
    'axes.facecolor':pal['card'],'grid.color':pal['line'],'svg.fonttype':'path'})
fig,axs=plt.subplots(2,2,figsize=(16,10))
fig.subplots_adjust(left=.065,right=.965,bottom=.10,top=.86,wspace=.25,hspace=.38)
fig.text(.045,.954,'STARLINK  /  SIGNAL RELATIONSHIPS',fontsize=21,weight='bold')
fig.text(.045,.914,'Whole-waveform hypotheses: drift, spectral structure, phase coupling and termination',fontsize=12,color=pal['muted'])
with np.load(artifact_path('global_signal_audit_arrays.npz'),allow_pickle=False) as z:
    f=z['receiver_frequency_hz'];d=z['smoothed_difference'];tt=z['time_s'];powers=z['band_power']
axis=-174000;scale=np.max(d)
ax=axs[0,0]
ax.plot(f/1000,d/scale,color=pal['primaryOnLight'],lw=1.8,label='Smoothed pre - post spectrum')
ax.plot(f/1000,np.interp(2*axis-f,f,d,left=np.nan,right=np.nan)/scale,color=pal['deep'],lw=1.3,alpha=.85,ls='--',label='Reflected about -174 kHz')
ax.axvline(axis/1000,color=pal['muted'],ls=':',lw=1)
ax.set(title='01 / Symmetry in power',xlabel='Receiver baseband offset (kHz)',ylabel='Event-associated spectral difference / peak',xlim=(-1200,1200),ylim=(-.14,1.13))
ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(alpha=.35)
drift=json.loads((artifact_path('wideband_drift_link.json')).read_text(encoding='utf8'))
with np.load(artifact_path('wideband_drift_link_arrays.npz'),allow_pickle=False) as z:
    held=z['heldout_blocks'];base=z['receiver_coordinates_heldout_correlations'];correct=z['moving_line_drift_removed_heldout_correlations'];opposite=z['opposite_drift_control_heldout_correlations']
with np.load(artifact_path('waterfall_detail_data.npz'),allow_pickle=False) as z:st=z['time_s']
order=np.argsort(st[held]);x=st[held][order]
ax=axs[0,1]
for values,label,col in [(correct,'Remove moving-line drift',pal['primaryOnLight']),(base,'Keep receiver coordinates',pal['secondary']),(opposite,'Reverse-drift control',pal['muted'])]:
    ax.plot(x,values[order],'.-',lw=.8,ms=4,label=f'{label}  ({values.mean():.3f})',color=col)
ax.set(title='02 / Broad spectrum stabilizes after drift removal',xlabel='Held-out block time (s)',ylabel='Shape agreement with training profile',ylim=(.74,1.01))
ax.legend(frameon=False,fontsize=9,loc='lower left');ax.grid(alpha=.35)
ax.text(.025,.96,'Narrow-line corridors excluded',transform=ax.transAxes,va='top',fontsize=10)
phase=json.loads((artifact_path('sideband_phase_coupling.json')).read_text(encoding='utf8'))
ax=axs[1,0];labels=['Receiver 0','CW freq.','CW phase','CW + 18k','-174 kHz'];y=np.arange(5)
scores=np.array([r['heldout_phase_shape_score'] for r in phase['results']]);threshold=np.array([r['phase_control_95pct'] for r in phase['results']])
ax.barh(y,scores,height=.45,color=pal['primaryOnLight'],label='Measured held-out score')
ax.scatter(threshold,y,marker='|',s=180,color=pal['planetOnLight'],label='95th percentile of phase controls')
ax.axvline(0,color=pal['muted'],lw=1);ax.set_yticks(y,labels,fontsize=10);ax.invert_yaxis()
ax.set(title='03 / No stable conjugate-sideband phase coupling',xlabel='Train-defined phase-profile agreement',xlim=(-.04,.08))
ax.legend(frameon=False,fontsize=9,loc='lower right');ax.grid(axis='x',alpha=.35)
ax=axs[1,1];mask=(tt>=13.05)&(tt<=14.25);baseline=(tt>=13.1)&(tt<=13.5)
for col,label,color in [(0,'Negative lobe',pal['primaryOnLight']),(1,'Positive lobe',pal['secondary']),(2,'Center region',pal['muted']),(5,'High-frequency edge',pal['deep'])]:
    value=10*np.log10(powers[:,col]/powers[baseline,col].mean())
    ax.plot(tt[mask],value[mask],lw=1.2,label=label,color=color)
ax.axvline(13.635,color=pal['ink'],ls=':',lw=1);ax.set(title='04 / Termination affects selected spectral regions',xlabel='Time (s)',ylabel='Band power relative to local pre-event mean (dB)')
ax.legend(frameon=False,fontsize=9,loc='lower left');ax.grid(alpha=.35)
fig.text(.045,.035,'Evidence supports linked frequency evolution; symmetry alone does not identify modulation or prove a shared transmitter.',fontsize=11,color=pal['ink'])
fig.savefig(artifact_path('global_hypothesis_tests.svg'));plt.close(fig)
tree=ET.parse(artifact_path('global_hypothesis_tests.svg'));svg=tree.getroot();svg.set('width','1536');svg.set('height','960')
logo=ET.parse(VI/'assets/horizontal-primary.svg').getroot();logo.set('x','905');logo.set('y','6');logo.set('width','224');logo.set('height','49')
if logo.get('data-color-mode')!='monochrome':
    for element in logo.iter():
        if element.get('data-stroke-role')=='planet':element.set('stroke',pal['planetOnLight'])
svg.append(logo);tree.write(artifact_path('global_hypothesis_tests.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('global_hypothesis_tests.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('global_hypothesis_tests.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
print('Saved global_hypothesis_tests.svg and .png')
