
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
import json, xml.etree.ElementTree as ET
import numpy as np
from scipy.io import wavfile
from scipy import fft, ndimage, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import resvg_py

OUT=experiment_output('E001', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
N=65536; K=4; win=signal.windows.hann(N,sym=False).astype(np.float32)
f=np.fft.fftshift(fft.fftfreq(N,1/fs)); times=[]; arr=[]
for pos in range(0,len(raw)-N*K+1,N*K):
    a=raw[pos:pos+N*K].astype(np.float32)/32768
    z=(a[:,0]+1j*a[:,1]).reshape(K,N)
    P=np.mean(abs(fft.fft(z*win,axis=-1))**2,axis=0)/(fs*np.sum(win**2))
    arr.append(np.fft.fftshift(P)); times.append((pos+N*K/2)/fs)
P=np.array(arr); t=np.array(times)
db=lambda x:10*np.log10(np.maximum(x,1e-30))
# Smooth local power estimate: weak narrow tones do not set the background.
bg=ndimage.uniform_filter1d(P,size=301,axis=1,mode='nearest')
contrast=db(P/bg)
flat=db(P)-np.median(db(P),axis=0,keepdims=True)
np.savez_compressed(artifact_path('waterfall_detail_data.npz'),frequency_hz=f,time_s=t,psd=P,local_contrast_db=contrast,temporal_residual_db=flat)
pal=json.loads(VI_ROOT / 'assets/palette-tokens.json'.read_text())[0]['colors']
plt.rcParams.update({'font.family':'Arial','font.size':10,'text.color':pal['ink'],'axes.labelcolor':pal['ink'],'axes.edgecolor':pal['line'],'xtick.color':pal['muted'],'ytick.color':pal['muted'],'figure.facecolor':pal['surface'],'axes.facecolor':pal['card']})
cmap=LinearSegmentedColormap.from_list('orbit',[pal['deep'],pal['primary'],pal['accent']])
fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
fig.suptitle('STARLINK  /  LOCAL WATERFALL STRUCTURE',x=.04,ha='left',fontsize=20,fontweight='bold')
# Max pooling preserves thin carriers when showing the entire band.
def panel(ax,lo,hi,title):
    mask=(f>=lo)&(f<=hi); ff=f[mask]; data=contrast[:,mask]
    pool=max(1,len(ff)//1600); usable=len(ff)//pool*pool
    data=data[:,:usable].reshape(len(t),-1,pool).max(axis=-1)
    ff=ff[:usable].reshape(-1,pool).mean(axis=-1)
    im=ax.pcolormesh(ff/1000,t,data,cmap=cmap,vmin=0,vmax=16,shading='auto',rasterized=True)
    ax.set(xlabel='Baseband offset (kHz)',ylabel='Time (s)',title=title)
    fig.colorbar(im,ax=ax,label='dB over local mean floor')
panel(axs[0,0],-1200000,1200000,'01 / Full band, local background removed')
panel(axs[0,1],-200000,200000,'02 / Center region at 36.6 Hz spectral bins')
panel(axs[1,0],200000,800000,'03 / Positive-frequency region')
panel(axs[1,1],-1000000,-400000,'04 / Negative-frequency region')
fig.savefig(artifact_path('waterfall_detail.svg')); plt.close(fig)
tree=ET.parse(artifact_path('waterfall_detail.svg')); svg=tree.getroot(); svg.set('width','1536'); svg.set('height','960')
logo=ET.parse(str(VI_ROOT / 'assets/horizontal-primary.svg')).getroot()
logo.set('x','918');logo.set('y','0');logo.set('width','220');logo.set('height','49')
for el in logo.iter():
    if el.get('data-stroke-role')=='planet':el.set('stroke',pal['planetOnLight'])
svg.append(logo);tree.write(artifact_path('waterfall_detail.svg'),encoding='utf8',xml_declaration=True)
(artifact_path('waterfall_detail.png')).write_bytes(resvg_py.svg_to_bytes(svg_string=(artifact_path('waterfall_detail.svg')).read_text(encoding='utf8'),width=1920,dpi=96))
# Peaks ranked by time-dependent prominence rather than whole-file averaging.
peaks=[]
for j in range(len(t)):
    ks,_=signal.find_peaks(contrast[j],height=11,distance=30)
    for k in ks:peaks.append({'time_s':float(t[j]),'frequency_hz':float(f[k]),'contrast_db':float(contrast[j,k])})
(artifact_path('waterfall_peaks.json')).write_text(json.dumps(peaks,indent=2),encoding='utf8')
print('Saved detail. Number of peaks over 11 dB:',len(peaks),flush=True)
print('Strongest noncentral peaks:',sorted([p for p in peaks if abs(p['frequency_hz']-13000)>2000 and abs(p['frequency_hz']+600000)>2000],key=lambda p:-p['contrast_db'])[:25],flush=True)
