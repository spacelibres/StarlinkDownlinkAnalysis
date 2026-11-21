"""Two-stage cyclic-autocorrelation search and frequency-resolved SCF.

Cycles are selected/refined only in calibration blocks. Whole held-out blocks
are checked with a joint max-statistic random-phase null. Existing WAV only.
"""

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
import argparse,json
import numpy as np
from scipy import signal,fft,ndimage
from scipy.io import wavfile

OUT=experiment_output('E005', '')
RF=400000.;CF=20000.;D=20;BLOCK=.2;LEN=int(RF*BLOCK);CLEN=int(CF*BLOCK)
LAGS=np.array([0,1,2,4,8,16,32,64,128,256,512])
BANDS={'negative_lobe':(-950000,-650000),'positive_lobe':(500000,750000),'center_control':(-100000,100000)}
SEED=20261128

def refine(products,train,coarse,lag):
    frequency=np.linspace(coarse-2.5,coarse+2.5,101)
    window=signal.windows.hann(CLEN,sym=False)
    values=signal.zoom_fft(products[:,lag]*window,[frequency[0],frequency[-1]],len(frequency),fs=CF,endpoint=True,axis=-1)/window.sum()
    absolute=np.arange(len(products))*BLOCK+LAGS[lag]/(2*RF)
    values*=np.exp(-2j*np.pi*absolute[:,None]*frequency[None,:])
    c=values[train];stat=abs(c.sum(axis=0))**2/np.maximum(np.sum(abs(c)**2,axis=0),1e-30)
    # A coherent-score ratio can favor the adjacent Hann sidelobe when its
    # random variance happens to be small. Localize by coherent amplitude in
    # the central coarse cell, then compute its coherent score separately.
    best=int(np.argmax(abs(c.sum(axis=0))**2))
    return float(frequency[best]),values[:,best],float(stat[best])

def scf_profile(x,alpha,blocks):
    nfft=1024;hop=512;window=signal.windows.hann(nfft,sym=False).astype(np.float32)
    starts=np.arange(0,LEN-nfft+1,hop)
    cross=np.zeros(nfft,complex);p1=np.zeros(nfft);p2=np.zeros(nfft)
    for b in blocks:
        absolute=b*BLOCK+np.arange(LEN)/RF
        rotation=np.exp(-1j*np.pi*alpha*absolute).astype(np.complex64)
        one=x[b]*rotation;two=x[b]*rotation.conj()
        u=fft.fft(one[starts[:,None]+np.arange(nfft)]*window,axis=1)
        v=fft.fft(two[starts[:,None]+np.arange(nfft)]*window,axis=1)
        cross+=(u*v.conj()).mean(axis=0);p1+=(abs(u)**2).mean(axis=0);p2+=(abs(v)**2).mean(axis=0)
    cross/=len(blocks);p1/=len(blocks);p2/=len(blocks)
    return np.fft.fftshift(cross),np.fft.fftshift(cross/np.sqrt(np.maximum(p1*p2,1e-30)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dechirp',action='store_true');parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    fs,raw=wavfile.read(RAW_WAV,mmap=True)
    assert fs==2400000
    count=len(raw)//int(fs*BLOCK);rng=np.random.default_rng(SEED)
    perm=rng.permutation(np.arange(65));train=perm[:32];test=perm[32:];post=np.arange(70,count)
    phase=rng.uniform(-np.pi,np.pi,(999,len(test)));postphase=rng.uniform(-np.pi,np.pi,(999,len(post)))
    coefficients=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
    reference=float(np.polyval(coefficients,6.5))
    buffers={};products={};arrays={};candidates=[];coarse_profiles={}
    for name,(lo,hi) in BANDS.items():
        h=signal.firwin(513,(hi-lo)/2,fs=fs,window=('kaiser',8)).astype(np.float32)
        x=np.empty((count,LEN),np.complex64);y=np.empty((count,len(LAGS),CLEN),np.complex64)
        for b in range(count):
            pos=round(b*BLOCK*fs);left=max(0,pos-4800);right=min(len(raw),pos+round(BLOCK*fs)+4800)
            a=raw[left:right].astype(np.float32)/32768;t=(left+np.arange(len(a)))/fs
            shifted=((a[:,0]+1j*a[:,1])*np.exp(-2j*np.pi*(lo+hi)/2*t)).astype(np.complex64)
            filtered=signal.fftconvolve(shifted,h,mode='same')
            dec=signal.resample_poly(filtered,1,6,window=('kaiser',8));offset=(pos-left)//6
            x[b]=dec[offset:offset+LEN]
            if args.dechirp:
                tm=b*BLOCK+np.arange(LEN)/RF
                ph=2*np.pi*(coefficients[0]*tm**3/3+coefficients[1]*tm*tm/2+(coefficients[2]-reference)*tm)
                x[b]*=np.exp(-1j*ph).astype(np.complex64)
            power=float(np.mean(abs(x[b])**2))
            for j,lag in enumerate(LAGS):
                prod=np.zeros(LEN,np.complex64)
                if lag:prod[:-lag]=x[b,lag:]*x[b,:-lag].conj()
                else:prod=x[b]*x[b].conj()
                prod-=prod[:LEN-lag if lag else LEN].mean()
                y[b,j]=signal.resample_poly(prod,1,D,window=('kaiser',8))[:CLEN]/max(power,1e-30)
            if b%20==0:print(name,'block',b,flush=True)
        buffers[name]=x;products[name]=y
        window=signal.windows.hann(CLEN,sym=False)
        spectrum=abs(fft.fft(y*window,axis=-1))**2/window.sum()**2
        freq=fft.fftfreq(CLEN,1/CF);select=(freq>=50)&(freq<=6000)
        cyclic_freq=freq[select]
        cal=spectrum[train].mean(axis=0)[:,select]
        baseline=ndimage.median_filter(cal,size=(1,31),mode='nearest')
        ratio=cal/np.maximum(baseline,1e-30);pooled=ratio.max(axis=0)
        peaks,_=signal.find_peaks(pooled,distance=4)
        chosen=sorted(peaks,key=lambda j:-pooled[j])[:5]
        coarse_profiles[name]=cal
        arrays[name+'_calibration_lag_cycle_power']=cal.astype(np.float32)
        arrays[name+'_heldout_lag_cycle_power']=spectrum[test].mean(axis=0)[:,select].astype(np.float32)
        arrays[name+'_post_lag_cycle_power']=spectrum[post].mean(axis=0)[:,select].astype(np.float32)
        for peak in chosen:
            lag_index=int(np.argmax(ratio[:,peak]));coarse=float(cyclic_freq[peak])
            refined,values,train_score=refine(y,train,coarse,lag_index)
            candidates.append({'band':name,'coarse_cycle_hz':coarse,'cycle_hz':refined,
                               'selection_kind':'blind_coarse_search',
                               'lag_samples_at400k':int(LAGS[lag_index]),'lag_index':lag_index,
                               'calibration_local_peak_ratio':float(pooled[peak]),
                               'calibration_coherent_score':train_score,'values':values})
        # Historical 750Hz is a separate, explicitly seeded hypothesis. Choose
        # its lag only on calibration data, include it in the same test family.
        kernel=window*np.exp(-2j*np.pi*750*np.arange(CLEN)/CF)/window.sum()
        prior=np.tensordot(y,kernel,axes=([-1],[0]))
        prior*=np.exp(-2j*np.pi*750*(np.arange(count)[:,None]*BLOCK+LAGS[None,:]/(2*RF)))
        c=prior[train];stat=abs(c.sum(axis=0))**2/np.maximum(np.sum(abs(c)**2,axis=0),1e-30)
        lag_index=int(np.argmax(stat))
        candidates.append({'band':name,'coarse_cycle_hz':750.,'cycle_hz':750.,'selection_kind':'historical_750Hz_prior',
                           'lag_samples_at400k':int(LAGS[lag_index]),'lag_index':lag_index,
                           'calibration_local_peak_ratio':None,'calibration_coherent_score':float(stat[lag_index]),'values':prior[:,lag_index]})
        print(name,'five blind candidates and one seeded hypothesis selected',flush=True)

    # All frozen candidates, including three seeded priors, are one family.
    # Preserve cross-candidate and
    # cross-band relationships by the same random phase for a whole block.
    V=np.stack([r['values'][test] for r in candidates])
    VP=np.stack([r['values'][post] for r in candidates])
    denom=np.maximum(np.sum(abs(V)**2,axis=1),1e-30);dp=np.maximum(np.sum(abs(VP)**2,axis=1),1e-30)
    actual=abs(V.sum(axis=1))**2/denom
    null=abs(np.exp(1j*phase)@V.T)**2/denom[None,:]
    pactual=abs(VP.sum(axis=1))**2/dp
    pnull=abs(np.exp(1j*postphase)@VP.T)**2/dp[None,:]
    nullmax=null.max(axis=1);pnullmax=pnull.max(axis=1)
    rows=[]
    for j,row in enumerate(candidates):
        key='candidate_'+str(j)
        arrays[key+'_complex_cyclic_coefficients']=row.pop('values').astype(np.complex64)
        row.update({'heldout_coherent_score':float(actual[j]),'heldout_global_max_null_95pct':float(np.quantile(nullmax,.95)),
                    'familywise_random_phase_p':float((1+np.sum(nullmax>=actual[j]))/1000),
                    'post_event_coherent_score':float(pactual[j]),'post_familywise_random_phase_p':float((1+np.sum(pnullmax>=pactual[j]))/1000)})
        row['passes_frozen_candidate_family_5pct']=row['familywise_random_phase_p']<=.05
        rows.append(row)
    # Frequency-resolved ordinary SCF for each band's strongest held-out frozen
    # candidate and for the historical 750-Hz hypothesis, kept explicitly apart.
    profiles=[]
    for name in BANDS:
        best=max([r for r in rows if r['band']==name and r['selection_kind']=='blind_coarse_search'],key=lambda r:r['heldout_coherent_score'])
        for label,alpha in [('frozen_candidate',best['cycle_hz']),('prior_750Hz',750.)]:
            for group,ids in [('calibration',train),('heldout',test),('post',post)]:
                rawscf,coherence=scf_profile(buffers[name],alpha,ids)
                arrays[name+'_'+label+'_'+group+'_scf']=rawscf.astype(np.complex64)
                arrays[name+'_'+label+'_'+group+'_spectral_coherence']=coherence.astype(np.complex64)
            profiles.append({'band':name,'label':label,'cycle_hz':alpha,
                             'candidate_family_pass':best['passes_frozen_candidate_family_5pct'] if label=='frozen_candidate' else None,
                             'selection_note':'Profile chosen for display after testing; its bin maxima are descriptive, not another independent significance test.'})
    arrays['cycle_frequency_hz']=cyclic_freq;arrays['lag_samples_at400k']=LAGS
    arrays['scf_local_frequency_hz']=fft.fftshift(fft.fftfreq(1024,1/RF))
    arrays['heldout_random_phases']=phase;arrays['post_random_phases']=postphase;arrays['joint_max_null']=nullmax
    report={'scope':'Original local WAV only; no new capture/transmission.',
            'method':'Ordinary second-order cyclic autocorrelation coarse search, train-only noninteger frequency refinement using CZT, frozen holdout tests with 999 shared whole-block random phase controls and familywise max statistic; Welch frequency-resolved SCF diagnostics.',
            'bands_hz':BANDS,'sample_rate_after_bandpass_hz':RF,'lag_product_rate_hz':CF,
            'ordinary_not_conjugate_or_higher_order':True,'calibration_seed':SEED,
            'frequency_drift_removed_using_prior_moving_line':bool(args.dechirp),
            'dechirp_reference_frequency_hz':reference if args.dechirp else None,
            'block_duration_s':BLOCK,'calibration_blocks':train.tolist(),'heldout_blocks':test.tolist(),'post_blocks':post.tolist(),
            'cycle_coarse_resolution_hz':5.,'refinement_grid_step_hz':.05,'scf_frequency_resolution_hz':RF/1024,
            'candidate_family_size':len(rows),'candidates':rows,'scf_profiles':profiles,
            'limitations':['Finite narrowband and low signal fraction can hide real cycles.','Ordinary SCF is second order; these results are not high-order cumulants or payload decoding.','One capture and one termination cannot identify source hardware.','Random-phase inference conditions on blockwise phase symmetry; 750-Hz prior is a separate diagnostic, not part of a claimed blind discovery.','The coarse search covers 50..6000Hz and selected lags only; absent candidates do not exclude higher symbol rates or other lags.']}
    prefix='subband_cyclostationarity'+('_'+args.suffix if args.suffix else '')
    (artifact_path(prefix+'.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(artifact_path(prefix+'_arrays.npz'),**arrays)
    print(json.dumps({'candidates':rows},indent=2),flush=True)

if __name__=='__main__':main()
