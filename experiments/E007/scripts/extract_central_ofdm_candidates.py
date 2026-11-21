"""Conditional narrowband OFDM symbols and partial repeated binary patterns.

Uses the tested half-subcarrier sync hypothesis. No bit mapping, FEC or payload
semantics are certified. All complex observations and erasures are retained.
"""

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
import csv,json
import numpy as np
from scipy.io import wavfile,loadmat

OUT=experiment_output('E007', '')
fs,raw=wavfile.read(RAW_WAV,mmap=True)
sync=json.loads((artifact_path('narrowband_sync_prefix_extended_test.json')).read_text(encoding='utf8'))
fold=json.loads((artifact_path('recovered_cyclic_waveform.json')).read_text(encoding='utf8'))
clock=1-fold['inherited_phase_drift_samples_s']/fs
with np.load(artifact_path('narrowband_sync_prefix_extended_observations.npz'),allow_pickle=False) as z:
    all_epochs=z['frame_epoch_s'];blocks=z['block_index'];projection=z['selected_projection']
score=abs(projection)**2
selected=[];groups=[]
for name,ids,limit in [('calibration',sync['calibration_blocks'],96),('heldout',sync['heldout_blocks'],96),('post',sync['post_heldout_blocks'],48)]:
    available=np.flatnonzero(np.isin(blocks,ids));chosen=available[np.argsort(score[available])[-limit:]]
    selected.extend(chosen);groups.extend([name]*len(chosen))
selected=np.array(selected);groups=np.array(groups)
epochs=all_epochs[selected]+sync['exact']['offset_from_power_guard_epoch_s']
phase_reference=projection[selected]/np.maximum(abs(projection[selected]),1e-20)
coeff=np.load(artifact_path('refined_track.npz'))['quadratic_coefficients']
signed=np.array([-4,-3,2,3,4,5]);unsigned=signed%1024
symbols=np.arange(1,302);TS=4.4e-6/clock;CP=(2/15*1e-6)/clock
carrier=(signed+.5)*234375*clock
def demod_frame(raw,epoch,fs,clock,coeff,carrier):
    symbols=np.arange(1,302);TS=4.4e-6/clock;CP=(2/15*1e-6)/clock
    centers=np.rint((epoch+symbols*TS+(CP+(TS-CP)/2))*fs).astype(int)
    n=centers[:,None]+np.arange(-4,4)[None,:]
    local=n/fs-epoch-symbols[:,None]*TS-CP
    t=n/fs
    phase=2*np.pi*(coeff[0]*t**3/3+coeff[1]*t*t/2+coeff[2]*t)
    values=raw[n].astype(np.float64)/32768
    y=(values[...,0]+1j*values[...,1])*np.exp(-1j*phase)
    A=np.exp(2j*np.pi*local[:,:,None]*carrier[None,None,:]);inverse=np.linalg.pinv(A)
    # Shared stationary-line coefficients, per-symbol data subcarriers free.
    B=np.stack([np.ones_like(t,dtype=complex),np.exp(2j*np.pi*13050*t-1j*phase),np.exp(-2j*np.pi*600000*t-1j*phase)],axis=-1)
    Q=B-np.einsum('snk,skq,sqp->snp',A,inverse,B)
    rhs=np.einsum('snp,sn->p',Q[1:].conj(),y[1:])
    gram=np.einsum('snp,snq->pq',Q[1:].conj(),B[1:])
    nuisance=np.linalg.pinv(gram)@rhs
    remaining=y-np.einsum('snp,p->sn',B,nuisance)
    estimates=np.einsum('skn,sn->sk',inverse,remaining)
    residual=remaining-np.einsum('snk,sk->sn',A,estimates)
    # The half-carrier shift is continuous in absolute time, while the OFDM
    # symbol basis restarts its local time each slot. Remove this deterministic
    # slot rotation, retaining only a frame-global phase shared with the prefix.
    estimates*=np.exp(-2j*np.pi*(234375*.5*clock)*(symbols*TS+CP))[:,None]
    return estimates,float(np.mean(abs(residual)**2)),float(np.median(np.linalg.cond(A)))

cube=[];fitnoise=[];conditions=[]
for i,epoch in enumerate(epochs):
    estimates,error,condition=demod_frame(raw,epoch,fs,clock,coeff,carrier)
    cube.append(estimates);fitnoise.append(error);conditions.append(condition)
cube=np.array(cube)
sss=loadmat(artifact_path('reference/qin/sssVec.mat'))['sssVecFull'].ravel()[unsigned]
cal=(groups=='calibration')
gain=np.mean(cube[cal,0]/phase_reference[cal,None]/sss[None,:],axis=0)
equalized=cube[:,1:]/phase_reference[:,None,None]/gain[None,None,:]

pn=np.zeros(32767,dtype=int);pn[:15]=[int(x) for x in '000000100000010']
for i in range(15,len(pn)):pn[i]=pn[i-14]^pn[i-15]
s=np.arange(2,302);ref=[]
for k in signed:
    ordinal=int(k)+512 if k<0 else int(k)+508;d=1020*(s-17)+ordinal
    rotation=(2*pn[(d+16327)%32767]+(1^pn[(d+16320)%32767])+1)%4
    ref.append(np.exp(.5j*np.pi*rotation))
ref=np.array(ref).T
deviation=equalized*ref.conj()[None,:,:]

# Nine two-cycle groups. Keep paired cycles together; guard two slots at each
# group boundary. Thus all 15 classes can remain represented in either split.
rng=np.random.default_rng(20261202);perm=rng.permutation(9);train_pairs=perm[:4];test_pairs=perm[4:]
cycle=(s-2)//15;pair=(cycle-2)//2;within=(s-2-30)%30
guard=(cycle>=2)&(within>=2)&(within<28)
train=guard&np.isin(pair,train_pairs);test=guard&np.isin(pair,test_pairs)
word_positions=(unsigned[None,:]-16*s[:,None])%60
words=[];rows=[];means_a=[];means_b=[]
for frame in range(len(selected)):
    a=[];b=[];ea=[];eb=[]
    # Interior four carriers have more margin from the capture edges.
    interior=np.isin(signed,[-3,2,3,4])
    for bit in range(60):
        aa=deviation[frame][(word_positions==bit)&train[:,None]&interior[None,:]]
        bb=deviation[frame][(word_positions==bit)&test[:,None]&interior[None,:]]
        a.append(aa.mean());b.append(bb.mean());ea.append(aa.std(ddof=1)/np.sqrt(len(aa))/np.sqrt(2));eb.append(bb.std(ddof=1)/np.sqrt(len(bb))/np.sqrt(2))
    a=np.array(a);b=np.array(b);ea=np.array(ea);eb=np.array(eb)
    # A train-only constant axis, retaining the global pi ambiguity.
    theta=.5*np.angle(np.sum(a*a));ar=(a*np.exp(-1j*theta)).real;br=(b*np.exp(-1j*theta)).real
    supported=(abs(ar)>3*ea)&(abs(br)>3*eb)&(ar*br>0)
    pattern=''.join(('1' if ar[j]<0 else '0') if supported[j] else '?' for j in range(60))
    coh=float(np.vdot(a,b).real/np.sqrt(np.vdot(a,a).real*np.vdot(b,b).real))
    words.append(pattern);means_a.append(a);means_b.append(b)
    rows.append({'source_prefix_row':int(selected[frame]),'group':str(groups[frame]),'epoch_s':float(epochs[frame]),
                 'prefix_projection_energy':float(score[selected[frame]]),'pair_split_word_coherence':coh,
                 'supported_relative_bits':int(supported.sum()),'relative_pattern_60':pattern,
                 'global_polarity_ambiguous':True,'message_crc_passed':False})
np.savez_compressed(artifact_path('central_ofdm_symbol_candidates.npz'),frame_epoch_s=epochs,group=groups,subcarrier_signed=signed,
                    symbols_including_sss=cube.astype(np.complex64),equalized_data=equalized.astype(np.complex64),
                    pn_reference=ref.astype(np.complex64),train_word_means=np.array(means_a).astype(np.complex64),test_word_means=np.array(means_b).astype(np.complex64),
                    source_prefix_rows=selected,train_cycle_pairs=train_pairs,test_cycle_pairs=test_pairs)
with (artifact_path('central_relative_patterns.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result={'scope':'Conditional six-carrier LS extraction; partial relative binary candidates, not decoded messages.',
        'grid_offset_from_line_hz':117187.5,'subcarriers_signed':signed.tolist(),'frames':len(rows),
        'clock_and_epoch_from_prior_sync_hypothesis':True,'seed':20261202,'train_cycle_pairs':train_pairs.tolist(),'test_cycle_pairs':test_pairs.tolist(),
        'median_design_condition':float(np.median(conditions)),'calibration_channel_gains_real':gain.real.tolist(),'calibration_channel_gains_imag':gain.imag.tolist(),
        'group_summaries':{name:{'frames':sum(r['group']==name for r in rows),'median_word_coherence':float(np.median([r['pair_split_word_coherence'] for r in rows if r['group']==name])),
                                 'supported_relative_bits':sum(r['supported_relative_bits'] for r in rows if r['group']==name),
                                 'max_supported_bits_in_a_frame':max(r['supported_relative_bits'] for r in rows if r['group']==name)} for name in ['calibration','heldout','post']},
        'validated_messages':0,'limitations':['Six-carrier time-domain model approximates narrowband receiver filtering and out-of-band leakage.','Prefix energy selection is not a payload quality certificate.','Channel gains use calibration SSS observations after a noisy per-frame prefix phase estimate.','Repeated-pattern and 3-sigma screens are exploratory, with many tested bits and no message-level FEC/CRC.','Global polarity and actual symbol-to-bit mapping remain unresolved.']}
(artifact_path('central_ofdm_candidates.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2),flush=True)
