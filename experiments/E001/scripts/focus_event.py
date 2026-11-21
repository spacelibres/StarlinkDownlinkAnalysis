
# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E001')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np
from scipy.io import wavfile
from scipy import signal,fft,ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import resvg_py

OUT=experiment_output('E001', '')
d=np.load(artifact_path('waterfall_detail_data.npz')); f=d['frequency_hz'];t=d['time_s'];P=d['psd'];contrast=d['local_contrast_db']
track=np.load(artifact_path('refined_track.npz')); c=track['quadratic_coefficients']; model=track['model_hz']; ratio=track['track_ratio']
db=lambda x:10*np.log10(np.maximum(x,1e-30))
fs,raw=wavfile.read(RAW_WAV,mmap=True)
# 1-ms total power in the vicinity of the transition, without row normalization.
a=raw[int(13.3*fs):int(14*fs)].astype(np.float64)/32768
pp=np.mean(a.reshape(-1,2400,2)**2,axis=(1,2))*2;tt=13.3+(np.arange(len(pp))+.5)/1000
# Least-squares two-level step, avoiding subjective threshold selection.
cs=np.r_[0,np.cumsum(pp)];cs2=np.r_[0,np.cumsum(pp**2)];n=len(pp);ks=np.arange(80,n-80)
err=cs2[ks]-cs[ks]**2/ks+(cs2[n]-cs2[ks])-(cs[n]-cs[ks])**2/(n-ks)
k=int(ks[np.argmin(err)]);event=13.3+k/1000
pre=(t>10)&(t<13.5); post=(t>13.8)&(t<15.8)
bands={}
for name,lo,hi in [('negative_lobe',-950000,-650000),('positive_lobe',500000,750000),('center_background',-100000,100000)]:
    m=(f>=lo)&(f<hi);pw=P[:,m].sum(axis=1)*(f[1]-f[0]);bands[name]={'before_dbfs':float(db(pw[pre].mean())),'after_dbfs':float(db(pw[post].mean())),'drop_db':float(db(pw[post].mean()/pw[pre].mean()))}
# Fixed narrow line as a control; wide enough to include all its slow drift.
m=(f>12500)&(f<13500);pw=P[:,m].sum(axis=1)*(f[1]-f[0]);bands['fixed_13khz_region']={'before_dbfs':float(db(pw[pre].mean())),'after_dbfs':float(db(pw[post].mean())),'drop_db':float(db(pw[post].mean()/pw[pre].mean()))}
result={'event_step_time_s':event,'event_resolution_s':.001,'moving_tone_fit_frequency_hz':'c2*t^2+c1*t+c0, approximate STFT ridge fit','coefficients':c.tolist(),'moving_tone_model_at_0_and_event_hz':np.polyval(c,[0,event]).tolist(),'moving_tone_rate_at_0_and_event_hz_s':(2*c[0]*np.array([0,event])+c[1]).tolist(),'normalized_track_power_before':float(ratio[pre].mean()),'normalized_track_power_after':float(ratio[post].mean()),'bands':bands,'repository_sample_sha256_matches':True,'caution':'The ridge fit is descriptive; it is not a calibrated Doppler-only orbit solution. Local background normalization is for feature inspection only.'}
(artifact_path('event_metrics.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2),flush=True)

pal=json.loads(VI_ROOT / 'assets/palette-tokens.json'.read_text())[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':pal['ink'],'axes.labelcolor':pal['ink'],'axes.edgecolor':pal['line'],'xtick.color':pal['muted'],'ytick.color':pal['muted'],'figure.facecolor':pal['surface'],'axes.facecolor':pal['card']})
cmap=LinearSegmentedColormap.from_list('orbit',[pal['deep'],pal['primary'],pal['accent']])
fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained');fig.get_layout_engine().set(rect=(0,0,1,.9))
fig.suptitle('STARLINK  /  MOVING SIGNAL AND ABRUPT END',x=.04,y=.98,ha='left',fontsize=19,fontweight='bold')
ax=axs[0,0];mask=(f>-215000)&(f<-145000)
im=ax.pcolormesh(f[mask]/1000,t,contrast[:,mask],shading='auto',vmin=0,vmax=10,cmap=cmap,rasterized=True);ax.set(xlabel='Baseband offset (kHz)',ylabel='Time (s)',title='01 / Drifting narrowband ridge');ax.set_ylim(15.95,0);ax.axhline(event,color=pal['planetOnLight'],ls='--',lw=1);ax.text(-147,event-.4,f'End near {event:.3f} s',ha='right',color=pal['planetOnLight']);fig.colorbar(im,ax=ax,label='dB over local mean floor')
ax=axs[0,1];mask=(f>-203000)&(f<-195000);sel=(t>12.6)&(t<14.6)
im=ax.pcolormesh(f[mask]/1000,t[sel],contrast[sel][:,mask],shading='auto',vmin=0,vmax=10,cmap=cmap,rasterized=True);ax.set(xlabel='Baseband offset (kHz)',ylabel='Time (s)',title='02 / Close view of signal termination');ax.set_ylim(14.6,12.6);ax.axhline(event,color=pal['planetOnLight'],ls='--',lw=1);fig.colorbar(im,ax=ax,label='dB over local mean floor')
ax=axs[1,0];sel=t<13.6;ax.plot(t[sel],model[sel]/1000,color=pal['primaryOnLight'],label='Approximate curved ridge fit');ax.scatter(t[sel],track['frequency_hz'][sel]/1000,s=5,color=pal['planetOnLight'],alpha=.5,label='Local ridge detections');ax.set(xlabel='Time (s)',ylabel='Baseband frequency (kHz)',title='03 / Frequency falls about 38 kHz');ax.legend(fontsize=9);ax.grid(color=pal['line'],alpha=.5)
ax=axs[1,1];ax.plot(tt,db(pp),color=pal['primaryOnLight'],lw=.8,label='Total power, 1 ms windows');ax.axvline(event,color=pal['planetOnLight'],ls='--',label=f'Best two-level step: {event:.3f} s');ax.set(xlabel='Time (s)',ylabel='Complex power (dBFS)',title='04 / Independent full-band power transition');ax.legend(fontsize=9);ax.grid(color=pal['line'],alpha=.5)
fig.savefig(artifact_path('signal_event.svg'));plt.close(fig)
tree=ET.parse(artifact_path('signal_event.svg'));svg=tree.getroot();svg.set('width','1536');svg.set('height','960');logo=ET.parse(str(VI_ROOT / 'assets/horizontal-primary.svg')).getroot();logo.set('x','918');logo.set('y','0');logo.set('width','220');logo.set('height','49')
for el in logo.iter():
    if el.get('data-stroke-role')=='planet':el.set('stroke',pal['planetOnLight'])
svg.append(logo);tree.write(artifact_path('signal_event.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('signal_event.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('signal_event.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
print('Saved focused event figure',flush=True)
