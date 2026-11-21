"""A held-cycle consistency check before combining complementary receivers."""

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
import json
import numpy as np
from scipy.io import loadmat

OUT=experiment_output('E003', '')
z=np.load(artifact_path('paired_tcode_observations.npz'),allow_pickle=False)
rot=loadmat(artifact_path('reference/qin/referenceTemplate.mat'))['referenceTemplateRotations']
symbols=np.arange(1,302);cycle=(symbols-2)//15
assignment=np.random.default_rng(20261120).permutation(np.arange(2,20));train=assignment[:9];test=assignment[9:]
data={}
for key,k in [('iq_2',537),('iq_3',525)]:
    index=list(z[key+'_indices']).index(k)
    ratio=z[key+'_sss_ratios'];loo=np.array([np.mean(np.delete(ratio,j,axis=0),axis=0) for j in range(len(ratio))])
    y=z[key+'_equalized'][:,:,index]*loo[:,index,None]
    deviation=y*np.exp(-.5j*np.pi*rot[k,symbols-1])[None,:]
    data[key]=[np.array([deviation[:,(symbols>31)&(((k-16*symbols)%60)==b)&np.isin(cycle,selected)].mean(axis=1) for b in range(k%4,60,4)]).T for selected in [train,test]]
A,At=data['iq_2'];B,Bt=data['iq_3']
def checked_score(b,bt):
    gain=np.vdot(b,A)/np.vdot(b,b);prediction=bt*gain
    return float(np.real(np.vdot(prediction,At))/np.sqrt(np.vdot(prediction,prediction).real*np.vdot(At,At).real))
actual=checked_score(B,Bt);rng=np.random.default_rng(20261122);permutations=[rng.permutation(15) for _ in range(200)]
null=[checked_score(B[:,p],Bt[:,p]) for p in permutations]
result={'test_signed_correlation':actual,'permuted_bit_position_control_max':max(null),'permuted_control_median':float(np.median(null)),'control_count':200,'split_seed':20261120,'control_seed':20261122,'training_cycle_indices':train.tolist(),'test_cycle_indices':test.tolist(),'control_permutations':[p.tolist() for p in permutations],'method':'Undo noisy SSS calibration; fit one common complex RX ratio using calibration cycles; test on separate complete cycles; bit indices matched through proposed 60-bit/16-shift model.','supports_joint_codeword':bool(actual>max(null)),'validated_message':False}
(artifact_path('paired_unknown_overlap_check.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print('Actual',actual,'control maximum',max(null),'joint codeword supported',result['supports_joint_codeword'])
