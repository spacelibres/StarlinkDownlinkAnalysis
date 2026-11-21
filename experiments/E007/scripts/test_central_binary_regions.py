"""Test PN-removed noncircularity, without assigning any bit values."""

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
import json
import numpy as np

OUT=experiment_output('E007', '')
with np.load(artifact_path('central_ofdm_symbol_candidates.npz'),allow_pickle=False) as z:
    values=z['equalized_data']*z['pn_reference'].conj()[None,:,:]
    groups=z['group'];carriers=z['subcarrier_signed']
keep=np.isin(carriers,[-3,2,3,4]);values=values[:,:,keep]
symbols=np.arange(2,302);times=symbols*4.4e-6
cal=np.flatnonzero(groups=='calibration');test=np.flatnonzero(groups=='heldout');post=np.flatnonzero(groups=='post')
regions=[(5,12),(5,20),(20,60),(60,150),(150,299)]
grid=np.arange(-2000,2001,25)
rng=np.random.default_rng(20261204);phase=rng.uniform(-np.pi,np.pi,(999,len(test)))
postphase=rng.uniform(-np.pi,np.pi,(999,len(post)));statistics=[];nulls=[];poststatistics=[];postnulls=[];rows=[]
for left,right in regions:
    mask=(symbols>=left)&(symbols<=right)
    y=values[:,mask];t=times[mask]
    square=y*y;normalization=np.mean(abs(y[cal])**2,axis=(0,1))
    mean=square[cal].mean(axis=0)/normalization[None,:]
    phasors=np.exp(-4j*np.pi*grid[:,None]*t[None,:])
    vectors=np.einsum('ds,sk->dk',phasors,mean)/len(t)
    candidate=int(np.argmax(np.linalg.norm(vectors,axis=1)));axis=vectors[candidate];axis/=np.linalg.norm(axis)
    perframe=np.einsum('s,fsk->fk',phasors[candidate],square)/len(t)/normalization[None,:]
    projection=perframe@axis.conj()
    v=projection[test];p=projection[post]
    measured=float(v.mean().real);after=float(p.mean().real)
    controls=(np.exp(2j*phase)@v/len(v)).real
    postcontrols=(np.exp(2j*postphase)@p/len(p)).real
    scale=controls.std(ddof=1);pscale=postcontrols.std(ddof=1)
    statistics.append(measured/scale);nulls.append(controls/scale)
    poststatistics.append(after/pscale);postnulls.append(postcontrols/pscale)
    rows.append({'symbol_range':[left,right],'calibration_selected_residual_cfo_hz':float(grid[candidate]),
                 'heldout_signed_noncircularity_score':measured/scale,'post_signed_noncircularity_score':after/pscale,
                 'calibration_search_at_cfo_boundary':bool(candidate in [0,len(grid)-1])})
joint=np.max(np.array(nulls),axis=0);pjoint=np.max(np.array(postnulls),axis=0)
for row,score,postscore in zip(rows,statistics,poststatistics):
    row['five_region_family_p']=float((1+np.sum(joint>=score))/1000)
    row['post_five_region_family_p']=float((1+np.sum(pjoint>=postscore))/1000)
result={'scope':'Conditional exploratory test of binary-axis structure after known PN derotation; no decoded bits.',
        'seed':20261204,'regions':rows,'prefix_overlap_avoided_by_starting_at_symbol5':True,
        'method':'Residual CFO and carrier-axis profile chosen on calibration frames only; heldout frame phase randomization, five-region max-statistic correction.',
        'limitations':['Chosen frames and phase/channel estimates inherit the prior sync model.','Positive noncircularity would support a binary-like projection, not a code, bit mapping or message.','A failed test can reflect low SNR, clock/phase mismatch, higher-order modulation or a wrong local receiver model.']}
(artifact_path('central_binary_region_test.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
