"""Signal-only boundary diagnostics and separately evaluated allocation models."""
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));import project_paths
from protocol_core import slot_start,FFT_SIZE
from layout_core import NONPILOT
from receiver_models import pn_reference

UNITS={3:8192,5:2048,9:2048,10:8192,12:4096,21:4096,27:2731,28:5462,37:2731,39:4096,68:2048}
BIT_MODELS={3:(8192,1),5:(2048,1),9:(2048,1),10:(16384,2),12:(8192,2),21:(8192,2),
            27:(8192,3),28:(16384,3),37:(8192,3),39:(16384,4)}


def coherence(values,window=32):
    values=np.asarray(values);size=len(values)//window
    if size<1:raise ValueError('No complete coherence window')
    blocks=values[:size*window].reshape(size,window)
    return np.abs(blocks.sum(axis=1))/np.maximum(np.abs(blocks).sum(axis=1),1e-30)


def data_field(iq,frame):
    """All 300 slots, same PN API as narrowband; no header/endpoint input."""
    slots=np.arange(2,302)
    starts=[slot_start(frame,int(s)) for s in slots]
    if min(starts)<0 or max(starts)+FFT_SIZE>len(iq):raise IndexError('Incomplete data FFT grid')
    raw=np.stack([np.fft.fft(np.asarray(iq[start:start+FFT_SIZE],dtype=np.complex128))[NONPILOT%FFT_SIZE] for start in starts])
    pn=np.column_stack([pn_reference(int(k),slots) for k in NONPILOT])
    wrong=np.column_stack([pn_reference(int(k),slots+1) for k in NONPILOT])
    return (raw*pn.conjugate()).ravel(),raw.ravel(),(raw*wrong.conjugate()).ravel()


def detect_suffix(profile,window=32,threshold=.45,block=16,need=.8,min_blocks=2):
    c=np.asarray(profile,dtype=float);nb=len(c)//block
    if nb<min_blocks:return {'status':'too_short','boundary_resource':None}
    fraction=np.mean(c[:nb*block].reshape(nb,block)>threshold,axis=1)
    good=fraction>=need;indices=np.flatnonzero(good)
    if not len(indices) or nb-1-indices[-1]>1:
        return {'status':'no_terminal_run','boundary_resource':None}
    end=int(indices[-1]);start=end
    while start>0 and good[start-1]:start-=1
    if end-start+1<min_blocks:return {'status':'too_short','boundary_resource':None,
                                    'good_blocks':end-start+1}
    first=start*block
    while first>0 and c[first-1]>threshold:first-=1
    before=c[max(0,first-block):first];after=c[first:min(len(c),first+2*block)]
    return {'status':'candidate','boundary_resource':int(first*window),'good_blocks':end-start+1,
            'trailing_bad_blocks':nb-1-end,'preceding_mean':float(before.mean()) if len(before) else None,
            'following_mean':float(after.mean()),'preceding_high_fraction':float(np.mean(before>threshold)) if len(before) else None,
            'local_mean_rise':float(after.mean()-before.mean()) if len(before) else None}


def least_squares_change(profile,window=32,min_windows=32):
    """Fixed one-change SSE model; a diagnostic, never a semantic boundary label."""
    x=np.asarray(profile,dtype=float);n=len(x)
    if n<2*min_windows:raise ValueError('Not enough profile points')
    prefix=np.r_[0,np.cumsum(x)];squares=np.r_[0,np.cumsum(x*x)]
    k=np.arange(min_windows,n-min_windows+1)
    costs=squares[k]-prefix[k]**2/k+squares[-1]-squares[k]-(prefix[-1]-prefix[k])**2/(n-k)
    j=int(k[np.argmin(costs)]);base=max(float(squares[-1]-prefix[-1]**2/n),0)
    score=max(0.,base-float(costs.min()))/base if base>1e-10 else 0.
    rise=float(x[j:].mean()-x[:j].mean())
    return {'boundary_resource':j*window,'left_mean':float(x[:j].mean()),'right_mean':float(x[j:].mean()),
            'mean_rise':rise,'sse_improvement_fraction':score,'min_windows':min_windows,
            'upward_candidate':bool(rise>=.15 and score>=.10)}


def upward_contrast(candidate):
    return bool(candidate['status']=='candidate' and candidate.get('local_mean_rise') is not None
                and candidate['local_mean_rise']>=.15 and candidate['preceding_high_fraction']<=.5)


def unit_identifiability(matrix,target,minimum_unit=1.):
    """Expose unit nullity and physical feasibility instead of rejecting a min-norm fit."""
    from scipy.optimize import linprog
    matrix=np.asarray(matrix,dtype=float);target=np.asarray(target,dtype=float)
    units,_,rank,_=np.linalg.lstsq(matrix,target,rcond=None)
    rows,columns=matrix.shape
    inequalities=np.r_[np.c_[matrix,-np.ones(rows)],np.c_[-matrix,-np.ones(rows)]]
    optimum=linprog(np.r_[np.zeros(columns),1.],A_ub=inequalities,b_ub=np.r_[target,-target],
                    bounds=[(minimum_unit,None)]*columns+[(0,None)],method='highs')
    if not optimum.success:raise RuntimeError('Unit-feasibility solver failed: '+optimum.message)
    return {'design_rank':int(rank),'free_unit_dimensions':columns-int(rank),
            'unconstrained_minimum_norm_units':units.tolist(),
            'fitted_rmse_resources':float(np.sqrt(np.mean((matrix@units-target)**2))),
            'minimum_unit':minimum_unit,'positive_unit_minimax_error':float(optimum.fun),
            'positive_unit_example':optimum.x[:-1].tolist()}


def predict_end(header_start,blocks,entries,joint_rounding=False):
    cursor=int(header_start)+114*int(blocks);allocations=[]
    for entry in entries:
        code=int(entry['mcs_candidate']);count=int(entry['quantity'])
        if code not in UNITS or count<=0:raise ValueError('Unknown unit or non-positive quantity')
        length=UNITS[code]*count
        if joint_rounding and code in BIT_MODELS:
            bits,alphabet_bits=BIT_MODELS[code];length=(bits*count+alphabet_bits-1)//alphabet_bits
        allocations.append({'code':code,'quantity':count,'start_resource':cursor,'end_resource':cursor+length,
                            'symbol_count':length})
        cursor+=length
    if header_start<0 or cursor>301200:raise ValueError('Predicted allocation outside data grid')
    return {'end_resource':cursor,'allocations':allocations,'remaining_resources':301200-cursor,
            'joint_rounding':joint_rounding}
