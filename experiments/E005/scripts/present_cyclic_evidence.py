"""Plot cyclic detections, SCF diagnostics and the measured chirp-lag law."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E005')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import resvg_py

OUT=experiment_output('E005', '');VI=VI_ROOT
pal=json.loads((VI/'assets/palette-tokens.json').read_text(encoding='utf8'))[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':pal['ink'],'axes.labelcolor':pal['ink'],
                    'axes.edgecolor':pal['line'],'xtick.color':pal['muted'],'ytick.color':pal['muted'],
                    'figure.facecolor':pal['surface'],'axes.facecolor':pal['card'],'axes.titleweight':'bold','svg.fonttype':'path'})
summary=json.loads((artifact_path('cyclic_evidence_summary.json')).read_text(encoding='utf8'))
drift=json.loads((artifact_path('cyclic_drift_consistency.json')).read_text(encoding='utf8'))
bands=['negative_lobe','positive_lobe','center_control'];labels=['Negative lobe','Positive lobe','Center region'];colors=[pal['primaryOnLight'],pal['secondary'],pal['muted']]
fig,axs=plt.subplots(2,2,figsize=(16,10));fig.subplots_adjust(left=.07,right=.96,bottom=.11,top=.86,hspace=.39,wspace=.25)
fig.text(.045,.952,'STARLINK  /  CYCLIC STRUCTURE',fontsize=22,weight='bold')
fig.text(.045,.91,'Independent subband tests and an explicit link between frequency drift and cyclic-frequency offsets',fontsize=11,color=pal['muted'])
with np.load(artifact_path('subband_cyclostationarity_dechirped_arrays.npz'),allow_pickle=False) as z:
    f=z['cycle_frequency_hz'];ax=axs[0,0]
    for band,label,color in zip(bands,labels,colors):
        p=z[band+'_calibration_lag_cycle_power'];baseline=ndimage.median_filter(p,size=(1,31),mode='nearest')
        ratio=(p/np.maximum(baseline,1e-30)).max(axis=0)
        ax.plot(f/1000,10*np.log10(ratio),lw=1,label=label,color=color)
    for cycle in np.arange(750,6001,750):ax.axvline(cycle/1000,lw=.6,ls=':',color=pal['line'])
    ax.set(title='01 / Blind calibration scan: 50-6000 Hz',xlabel='Cycle frequency (kHz)',ylabel='Local cyclic power ratio, max over 11 lags (dB)',xlim=(.05,6))
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.25,color=pal['line'])
    ax=axs[1,0];freq=z['scf_local_frequency_hz']
    for band,label,color,width in zip(bands,labels,colors,[110000,85000,60000]):
        keep=abs(freq)<=width
        for group,style in [('heldout','-'),('post','--')]:
            value=z[band+'_prior_750Hz_'+group+'_spectral_coherence']
            smooth=ndimage.uniform_filter1d(value.real,41)+1j*ndimage.uniform_filter1d(value.imag,41)
            ax.plot(freq[keep]/1000,abs(smooth[keep]),ls=style,lw=1.2,color=color,label=label+(' / held-out' if group=='heldout' else ' / post-event'))
    ax.set(title='03 / Ordinary SCF at 750 Hz: diagnostic profiles',xlabel='Frequency relative to each subband center (kHz)',ylabel='Smoothed spectral-coherence magnitude',ylim=(-.003,.10))
    ax.legend(frameon=False,fontsize=8,ncol=2,loc='upper right');ax.grid(alpha=.25,color=pal['line'])
ax=axs[0,1]
for band,label,color in zip(bands,labels,colors):
    rows=[r for r in summary['candidates'] if r['coordinate_run']=='dechirped' and r['band']==band]
    blind=[r for r in rows if r['selection_kind']=='blind_coarse_search'];prior=[r for r in rows if r['selection_kind']!='blind_coarse_search']
    ax.scatter([r['cycle_hz']/1000 for r in blind],[r['heldout_coherent_score'] for r in blind],s=35,color=color,label=label)
    ax.scatter([r['cycle_hz']/1000 for r in prior],[r['heldout_coherent_score'] for r in prior],s=110,marker='*',color=color,edgecolor=pal['ink'],lw=.4)
ax.axhline(summary['joint_36_max_null_95pct'],color=pal['ink'],ls='--',lw=1,label='95% joint max-null threshold')
ax.text(.03,.96,'Stars: historical 750 Hz prior; circles: blind candidates',transform=ax.transAxes,va='top',fontsize=9)
ax.set(title='02 / Frozen held-out tests: 36-test correction',xlabel='Cycle frequency (kHz)',ylabel='Whole-block coherent statistic',xlim=(.05,6),ylim=(0,35))
ax.legend(frameon=False,fontsize=9,loc='upper right');ax.grid(alpha=.25,color=pal['line'])
ax=axs[1,1];tau=np.array([r['lag_s'] for r in drift['rows']]);offset=np.array([r['observed_raw_offset_hz'] for r in drift['rows']]);after=np.array([r['dechirped_offset_hz'] for r in drift['rows']])
line=np.linspace(0,.00067,100)
ax.plot(line*1e6,line*drift['prior_line_mean_frequency_rate_hz_per_s'],color=pal['deep'],lw=1.5,label='Prediction from prior thin-line drift')
ax.scatter(tau*1e6,offset,s=42,color=pal['primaryOnLight'],label='Measured before correction')
ax.scatter(tau*1e6,after,s=65,marker='+',color=pal['secondary'],label='After drift correction')
ax.set(title=r'04 / Lag-dependent offset: $\Delta\alpha\simeq\tau\dot f$',xlabel='Correlation lag (microseconds)',ylabel='Offset from nearest 750 Hz harmonic (Hz)',ylim=(-2.1,.28))
ax.legend(frameon=False,fontsize=9,loc='lower left');ax.grid(alpha=.25,color=pal['line'])
fig.text(.045,.042,'Shared cyclic timing and frequency evolution are supported. Symbol rate, hardware origin and payload contents remain unresolved.',fontsize=11)
fig.savefig(artifact_path('subband_cyclic_structure.svg'));plt.close(fig)
tree=ET.parse(artifact_path('subband_cyclic_structure.svg'));svg=tree.getroot();svg.set('width','1536');svg.set('height','960')
logo=ET.parse(VI/'assets/horizontal-primary.svg').getroot();logo.set('x','905');logo.set('y','6');logo.set('width','224');logo.set('height','49')
if logo.get('data-color-mode')!='monochrome':
    for element in logo.iter():
        if element.get('data-stroke-role')=='planet':element.set('stroke',pal['planetOnLight'])
svg.append(logo);tree.write(artifact_path('subband_cyclic_structure.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('subband_cyclic_structure.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('subband_cyclic_structure.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
print('Saved subband_cyclic_structure.svg and .png')
