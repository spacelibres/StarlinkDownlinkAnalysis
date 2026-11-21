"""Ideal-erasure bounds: FEC/CRC cannot reconstruct unobserved degrees of freedom."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'learning'))
from project_paths import LEO_SNAPSHOT,experiment_output,require_run_context
from educational_fec_demo import encode_block,crc8

def encoder_matrix(blocks):
    output=np.zeros((114*blocks,32*blocks),dtype=np.uint8)
    for block in range(blocks):
        for column in range(32):
            bits=np.zeros(32,dtype=np.uint8);bits[column]=1
            output[block*114:(block+1)*114,block*32+column]=encode_block(bits)
    return output

def gf2_rank(matrix):
    a=np.array(matrix,dtype=np.uint8).copy();row=0
    for col in range(a.shape[1]):
        candidates=np.flatnonzero(a[row:,col])
        if not len(candidates):continue
        pivot=row+int(candidates[0]);a[[row,pivot]]=a[[pivot,row]]
        for other in range(a.shape[0]):
            if other!=row and a[other,col]:a[other]^=a[row]
        row+=1
        if row==a.shape[0]:break
    return row

def crc_constraints(inputs):
    # CRC is affine with init FF. Its constant changes no rank/dependency.
    count=inputs-8;matrix=np.zeros((8,inputs),dtype=np.uint8)
    zero=crc8(np.zeros(count,dtype=np.uint8))
    for column in range(count):
        basis=np.zeros(count,dtype=np.uint8);basis[column]=1;value=crc8(basis)^zero
        matrix[:,column]=[(value>>(7-j))&1 for j in range(8)]
    matrix[:,count:]=np.eye(8,dtype=np.uint8)
    return matrix

def describe(positions,blocks):
    matrix=encoder_matrix(blocks);observed=matrix[positions]
    augmented=np.vstack([observed,crc_constraints(32*blocks)])
    prefix=np.eye(32*blocks,dtype=np.uint8)[:16]
    withprefix=np.vstack([augmented,prefix]);rank=gf2_rank(withprefix)
    identifiable=[]
    for bit in range(32*blocks):
        e=np.zeros(32*blocks,dtype=np.uint8);e[bit]=1
        if gf2_rank(np.vstack([withprefix,e]))==rank:identifiable.append(bit)
    return {'observed_coded_positions':positions,'retained_input_bits':32*blocks,
            'ideal_coded_constraint_rank':gf2_rank(observed),'rank_with_frozen_crc':gf2_rank(augmented),
            'free_binary_dimensions_even_with_crc_and_known_prefix':32*blocks-rank,
            'individually_identifiable_inputs_if_bits_phase_and_prefix_were_known':identifiable,
            'rank_with_all_coded_resources':gf2_rank(matrix)}

def main():
    require_run_context();out=experiment_output('E008');out.mkdir(parents=True,exist_ok=True)
    nonpilot=[k for k in range(-512,512) if k not in [-2,-1,0,1] and not(-496<=k<=-489 or 488<=k<=495)]
    header=nonpilot[:228];windows=[]
    for dataset in ['DSX01-S05','paired_RX']:
        metadata=json.loads((LEO_SNAPSHOT/'reports/2026_10_18_dsx01_pilot_clock'/dataset/'selection.json').read_text(encoding='utf8'))
        for window in metadata['windows']:
            cfo=window['winner']['fractional_tracking_cfo_hz'];fs=metadata['sample_rate_hz']
            carriers=[k for k in range(520,545) if abs((k-531.5)*234375+cfo)<fs*.49 and not 528<=k<=535]
            positions=[header.index(k-1024) for k in carriers if k-1024 in header]
            windows.append({'dataset':dataset,'window':'iq_'+str(window['number']),**describe(positions,2)})
    report={'scope':'Conditional published header encoder/resource map, ideal error-free bits, known polarity and known first16; optimistic upper bound, not an executed message decoder.',
            'npz_header1_windows':windows,
            'optimistic_nonsimultaneous_npz_union_not_a_valid_codeword':describe([13,14,15,16,17],2),
            'wav_header1_center_capture':describe([],2),
            'validated_unknown_bits':0,'validated_messages':0,
            'conclusion':'CRC adds at most eight binary constraints. Missing early-header resources leave unconstrained information even under optimistic known-prefix assumptions; do not fill erasures with Viterbi defaults.',
            'limitations':['The published field layout is conditional on matching the observed transmitter format.',
                           'Combining different frames or receiver captures does not produce a simultaneous codeword.',
                           'Narrowband complex observations are noisier and phase-ambiguous, so actual identifiability cannot exceed this bound.',
                           'This bound concerns the published first PDU header, not all conceivable beacon formats or all physically observable waveform parameters.']}
    (out/'code_observability_bound.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
