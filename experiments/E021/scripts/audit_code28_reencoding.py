"""Independently re-encode the recovered cyclic generator using FFT convolution."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, SYSTEMATIC, coordinates
from validate_code21 import digest
from validate_code37 import constellation_for

RUN = ROOT/'runs/v019-code28-reproduction'


def fft_binary_parities(u, p):
    """Counts are small exact integers before reducing convolution over GF(2)."""
    fu, fp = np.fft.rfft(u, axis=-1), np.fft.rfft(p, axis=-1)
    sums = np.fft.irfft(np.einsum('uif,ijf->ujf', fu, fp), n=u.shape[-1], axis=-1)
    rounded = np.rint(sums)
    max_residual = float(np.max(abs(sums-rounded)))
    if max_residual >= 1e-7:
        raise RuntimeError('FFT coefficient too far from an integer')
    return (rounded.astype(np.int32) & 1).astype(np.uint8).reshape(len(u),-1), max_residual


def remodulate(bits, mask, samples, points):
    indices = bits.reshape(len(bits), SYMBOLS,3)[:,mask] @ np.array([4,2,1],dtype=np.uint8)
    y = samples[:,mask]
    return (np.sum(abs(y-points[indices])**2,axis=1)/np.sum(abs(y)**2,axis=1)).tolist()


def main():
    require_run_context()
    out = experiment_output('E021')
    out.mkdir(parents=True, exist_ok=True)
    folder = RUN/'artifacts/experiments/E021/results'
    numeric = folder/'code28_reproduction.npz'
    meta = folder/'code28_reproduction.json'
    manifest_path = RUN/'manifest.json'
    sync = json.loads(manifest_path.read_text(encoding='utf-8'))
    numeric_hash, metadata_hash = digest(numeric), digest(meta)
    if (numeric_hash != sync['changed_artifacts']['experiments/E021/results/code28_reproduction.npz']['sha256']
            or metadata_hash != sync['changed_artifacts']['experiments/E021/results/code28_reproduction.json']['sha256']):
        raise RuntimeError('Upstream reproduction evidence changed')
    report = json.loads(meta.read_text(encoding='utf-8'))
    with np.load(numeric,allow_pickle=False) as data:
        final = data['final_bits']
        hard = data['primary_hard_bits']
        matrix = data['generator_seed_bits'].reshape(9,7,LIFT)
        mask = data['coordinate_mask']
        samples = data['primary_symbols']
        original = data['original_hard_bits']
    u, v, _ = coordinates(final)
    base, _v0, _ = coordinates(hard)
    candidates = {
        'accepted':u,
        'initial':base,
        'original_receiver':coordinates(original)[0],
        'observed_middle_as_systematic':final[:,:MOTHER_K],
        'virtual_zeros_at_end':np.c_[final[:,:SYSTEMATIC],np.zeros((len(final),OMITTED),np.uint8)],
        'global_complement':coordinates(1-final)[0],
    }
    words = len(final)
    results = {}
    numerical = {}
    for key, blocks in candidates.items():
        parity, residual = fft_binary_parities(blocks.reshape(words,9,LIFT),matrix)
        target = {'initial':hard[:,MOTHER_K:MOTHER_N],'original_receiver':original[:,MOTHER_K:MOTHER_N],
                  'global_complement':1-v}.get(key,v)
        errors = np.count_nonzero(parity!=target,axis=1)
        results[key] = {'exact_words':int(np.count_nonzero(errors==0)),
                        'differences_per_word':errors.tolist()}
        numerical[key] = residual
    if results['accepted']['exact_words']!=report['final_exact_words'] or results['initial']['exact_words']!=report['initial_exact_words']:
        raise RuntimeError('Independent parity evaluation disagrees')
    flip = final.copy()
    flip[:,MOTHER_K:MOTHER_N:16] ^= 1
    fully_coded = mask.reshape(SYMBOLS,3).all(axis=1)
    points = constellation_for(2,False,True)
    output = {'schema':'starlink-code28-independent-reencoding/v1','source_sha256':numeric_hash,'source_json_sha256':metadata_hash,
              'source_manifest_sha256':digest(manifest_path), 'exact_parity_bits_tested':v.size,
              'fft_rounding_residual':numerical, 'results':results,
              'unknown_positions_unchanged':bool(np.array_equal(final[:,~mask],hard[:,~mask])),
              'core_nmse':remodulate(final,fully_coded,samples,points),
              'parity_flipped_nmse':remodulate(flip,fully_coded,samples,points)}
    (out/'code28_independent_audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('FFT verified',v.size,'parity coordinates; exact words',
          {key:row['exact_words'] for key,row in results.items()},flush=True)


if __name__=='__main__':
    main()
