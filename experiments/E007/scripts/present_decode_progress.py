"""Show acquired structure without portraying complex estimates as decoded bits."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E007')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import resvg_py

OUT=experiment_output('E007', '');VI=VI_ROOT
p=json.loads((VI/'assets/palette-tokens.json').read_text(encoding='utf8'))[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':11,'text.color':p['ink'],'axes.labelcolor':p['ink'],
                    'axes.edgecolor':p['line'],'xtick.color':p['muted'],'ytick.color':p['muted'],
                    'figure.facecolor':p['surface'],'axes.facecolor':p['card'],'axes.titleweight':'bold','svg.fonttype':'path'})
fig,axs=plt.subplots(2,2,figsize=(16,10));fig.subplots_adjust(left=.07,right=.96,bottom=.11,top=.86,hspace=.38,wspace=.24)
fig.text(.045,.952,'STARLINK  /  ACQUISITION TO SYMBOLS',fontsize=21,weight='bold')
fig.text(.045,.91,'Recovered cycle profiles, a central-gutter model, known-prefix evidence and conditional data estimates',fontsize=11,color=p['muted'])
with np.load(artifact_path('recovered_cyclic_waveform.npz'),allow_pickle=False) as z:
    phase=z['phase_us'];ax=axs[0,0]
    for name,label,color in [('negative_lobe','Negative lobe',p['primaryOnLight']),('positive_lobe','Positive lobe',p['secondary']),('center_control','Center region',p['deep'])]:
        ax.plot(phase,z[name+'_heldout_profile'],color=color,lw=1.4,label=label)
    ax.axhline(1,color=p['muted'],ls=':',lw=1);ax.set(xlim=(620,680),title='01 / Repeatable cycle boundary structure',xlabel='Phase within the fitted cycle (microseconds)',ylabel='Held-out power / band mean')
    ax.legend(frameon=False,fontsize=9);ax.grid(color=p['line'],alpha=.3)
with np.load(artifact_path('central_gutter_grid_test_arrays.npz'),allow_pickle=False) as z:
    x=z['offset_from_moving_line_hz']/1000;ax=axs[0,1]
    ax.plot(x,z['heldout_excess_ratio'],color=p['primaryOnLight'],lw=1.6,label='Held-out spectral difference')
    ax.plot(x,z['nulls_4_prediction'],color=p['deep'],lw=1.4,ls='--',label='Four-null model: corr. 0.955')
    ax.plot(x,z['nulls_2_prediction'],color=p['muted'],lw=1,alpha=.7,label='Two-null alternative')
    ax.set(title='02 / Conditional OFDM frequency grid',xlabel='Frequency relative to moving line (kHz)',ylabel='Spectral excess / post-event reference')
    ax.legend(frameon=False,fontsize=9);ax.grid(color=p['line'],alpha=.3)
sync=json.loads((artifact_path('narrowband_sync_prefix_extended_test.json')).read_text(encoding='utf8'))
ax=axs[1,0]
one=[r['heldout_whitened_energy'] for r in sync['randomized_controls']];two=[r['heldout_whitened_energy'] for r in sync['structure_preserving_controls']]
ax.hist(one,bins=32,density=True,histtype='step',color=p['muted'],lw=1.4,label='999 spectrum-preserving controls')
ax.hist(two,bins=24,density=True,histtype='step',color=p['secondary'],lw=1.4,label='384 structure-preserving controls')
ax.axvline(sync['exact']['heldout_whitened_energy'],color=p['primaryOnLight'],lw=2,label='Published PSS+SSS reference')
ax.set(title='03 / Known-prefix specificity on held-out blocks',xlabel='Whitened matched-prefix energy',ylabel='Control density');ax.set_ylim(0,ax.get_ylim()[1]*1.35)
ax.legend(frameon=False,fontsize=9,loc='upper left');ax.grid(color=p['line'],alpha=.3)
with np.load(artifact_path('central_ofdm_symbol_candidates.npz'),allow_pickle=False) as z:
    v=z['symbols_including_sss'][:,1:];group=z['group'];k=z['subcarrier_signed']
    before=np.mean(abs(v[group=='heldout'])**2,axis=(0,1));after=np.mean(abs(v[group=='post'])**2,axis=(0,1))
    ax=axs[1,1];ax.bar(np.arange(len(k)),before/after,color=p['primaryOnLight'],width=.6)
    ax.set_xticks(np.arange(len(k)),[str(i) for i in k]);ax.axhline(1,color=p['deep'],ls='--',lw=1)
    ax.set(title='04 / Extracted data positions remain noise limited',xlabel='Signed subcarrier index under the fitted grid',ylabel='Signal-region / post-event mean estimate power',ylim=(0,1.65))
    ax.text(.04,.94,'192 signal-region cycles + 48 post-event controls\nUnknown binary-pattern tests: not passed',transform=ax.transAxes,va='top',fontsize=10)
    ax.grid(axis='y',color=p['line'],alpha=.3)
fig.text(.045,.04,'Prefix matching and symbol extraction are supported processing stages; no unknown payload bit or message CRC is certified.',fontsize=11)
fig.savefig(artifact_path('decode_progress.svg'));plt.close(fig)
tree=ET.parse(artifact_path('decode_progress.svg'));svg=tree.getroot();svg.set('width','1536');svg.set('height','960')
logo=ET.parse(VI/'assets/horizontal-primary.svg').getroot();logo.set('x','905');logo.set('y','6');logo.set('width','224');logo.set('height','49')
if logo.get('data-color-mode')!='monochrome':
    for element in logo.iter():
        if element.get('data-stroke-role')=='planet':element.set('stroke',p['planetOnLight'])
svg.append(logo);tree.write(artifact_path('decode_progress.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('decode_progress.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('decode_progress.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
print('Saved decode_progress.svg and .png')
