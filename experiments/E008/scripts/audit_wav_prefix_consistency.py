"""Split known-prefix phase consistency, conditioned on the prior WAV model."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy import fft
from scipy.io import loadmat

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT))
from project_paths import legacy_artifact,experiment_output,require_run_context
from receiver_models import split_prefix_basis

def main():
    require_run_context();out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(legacy_artifact('narrowband_sync_prefix_extended_test.json').read_text(encoding='utf8'))
    with np.load(legacy_artifact('narrowband_sync_prefix_extended_observations.npz')) as z:
        x=z['prefix_iq'].astype(complex);blocks=z['block_index'];epochs=z['frame_epoch_s'];relative=z['reference_relative_sample']
    noise=np.isin(blocks,metadata['post_noise_calibration_blocks'])
    covariance=x[noise].T@x[noise].conj()/noise.sum();e,u=np.linalg.eigh(covariance)
    floor=np.trace(covariance).real/len(relative)*.001
    w=(u/np.sqrt(np.maximum(e,floor))[None,:])@u.conj().T;white=x@w.T
    register=[0,0,1,1,0,1,0];bits=[]
    for _ in range(127):
        value=register[2]^register[6];register=[value]+register[:-1];bits.append(value)
    segment=np.exp(-1j*np.pi/4-.5j*np.pi*np.cumsum(2*np.r_[0,bits[::-1]]-1))
    pss=np.r_[-segment,np.tile(segment,7)];pss=np.r_[-pss[-32:],pss]
    sss_frequency=loadmat(legacy_artifact('reference/qin/sssVec.mat'))['sssVecFull'].ravel()
    taper=np.ones(1056);taper[:16]=np.arange(16)/16;taper[-16:]=np.arange(16,0,-1)/16
    count=80000;fullrate=240000000.;time=(np.arange(count)-count//2)/fullrate
    frequencies=fft.fftfreq(count,1/fullrate);coeff=np.load(legacy_artifact('refined_track.npz'))['quadratic_coefficients']
    calibration=np.isin(blocks,metadata['calibration_blocks'])
    center=-float(np.polyval(coeff,epochs[calibration].mean()))
    response=np.clip((1200000-abs(frequencies-center))/150000,0,1)
    shifted=metadata['exact']['carrier_offset_from_moving_line_hz'];offset=metadata['exact']['offset_from_power_guard_epoch_s']
    narrowed_time=(np.arange(count//25)-count//50)/(fullrate/25)
    def vector(wave,start):
        padded=np.zeros(count,complex);padded[count//2+start:count//2+start+len(wave)]=wave
        narrowed=fft.ifft(fft.fft(padded*np.exp(2j*np.pi*shifted*time))*response)[::25]
        at=relative/2400000-offset
        value=np.interp(at,narrowed_time,narrowed.real)+1j*np.interp(at,narrowed_time,narrowed.imag)
        return w@(value-value.mean())
    p=vector(pss*taper,0)
    def sss_vector(frequency):
        s=32*fft.ifft(frequency);return vector(np.r_[s[-32:],s]*taper,1056)
    s=sss_vector(sss_frequency)
    seed=2026112009;rng=np.random.default_rng(seed);heldids=metadata['heldout_blocks'];postids=metadata['post_heldout_blocks']
    phases=rng.uniform(-np.pi,np.pi,(999,len(heldids)))
    def evaluate(s):
        a,b,rotation=split_prefix_basis(p,s);pa=white@a.conj();pb=white@b.conj()
        cross=pa*pb.conj()*rotation;bias=cross[noise].mean()
        perblock=np.array([cross[blocks==i].mean()-bias for i in heldids])
        controls=(np.exp(1j*phases)@perblock/len(perblock)).real
        score=float(perblock.mean().real/max(controls.std(ddof=1),1e-30))
        groups={}
        for name,ids in [('calibration',metadata['calibration_blocks']),('heldout',heldids),('post',postids)]:
            use=np.isin(blocks,ids);c=cross[use].mean()-bias
            groups[name]={'frames':int(use.sum()),'noise_adjusted_cross_real':float(c.real),'noise_adjusted_cross_imag':float(c.imag),
                          'raw_normalized_cross_magnitude':float(abs((pa[use]*pb[use].conj()).mean())/np.sqrt(np.mean(abs(pa[use])**2)*np.mean(abs(pb[use])**2)))}
        return {'groups':groups,'heldout_phase_score':score,'whole_block_phase_p':(1+int(np.sum(controls>=perblock.mean().real)))/1000},pa,pb
    exact,pa,pb=evaluate(s);wrong=[];loaded=np.flatnonzero(abs(sss_frequency)>0)
    for _ in range(199):
        frequency=sss_frequency.copy();frequency[loaded]=frequency[rng.permutation(loaded)]
        check,_,_=evaluate(sss_vector(frequency));wrong.append(check['groups']['heldout']['noise_adjusted_cross_real'])
    rank=(1+int(np.sum(np.array(wrong)>=exact['groups']['heldout']['noise_adjusted_cross_real'])))/200
    # Detection quality gate: calibrate PSS-only power threshold, predict the
    # orthogonal SSS direction on all held blocks; do not select with full prefix.
    threshold=float(np.quantile(abs(pa[calibration])**2,.8));quality={}
    for name,ids in [('heldout',heldids),('post',postids)]:
        use=np.isin(blocks,ids);strong=abs(pa)**2>=threshold
        quality[name]={'pss_only_threshold':threshold,'strong_frames':int(np.sum(use&strong)),
                       'orthogonal_sss_mean_energy_pss_strong':float(np.mean(abs(pb[use&strong])**2)),
                       'orthogonal_sss_mean_energy_pss_weak':float(np.mean(abs(pb[use&~strong])**2))}
    report={'scope':'Conditional PSS/SSS internal phase test using inherited WAV clock, grid, timing and whitening blocks.',
            'seed':seed,'exact':exact,'wrong_sss_controls':199,'wrong_sss_cross_real':wrong,'exact_wrong_sss_rank_fraction':rank,
            'quality_prediction':quality,'validated_unknown_bits':0,'validated_messages':0,
            'limitations':['Prior timing/grid and candidate family used this capture; this is conditional follow-up, not a new untouched validation dataset.',
                           'Noise-orthogonal directions are not independent RF measurements; whitening and finite receiver response remain assumptions.',
                           'Wrong SSS controls preserve loaded-bin magnitudes and locations but do not exhaust all alternative signals.',
                           'Positive internal coherence would support a common known-prefix model, not independently certified per-frame payload phase.',
                           'Whole-block phase inference assumes rotation symmetry; post event has few blocks.']}
    (out/'wav_prefix_consistency.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    np.savez_compressed(out/'wav_split_prefix_observations.npz',pss_projection=pa.astype(np.complex64),sss_orthogonal_projection=pb.astype(np.complex64),block_index=blocks)
    print(json.dumps({'exact':exact,'wrong_sss_rank_fraction':rank,'quality':quality},indent=2))

if __name__=='__main__':main()
