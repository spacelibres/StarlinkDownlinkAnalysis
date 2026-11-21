"""Compare receive branches and reproduce full Code 28 shortening."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'learning')]
from project_paths import experiment_output, legacy_artifact, require_run_context
from code5_core import LOADED, fft_grid, reference_gain, recover_generator
from code21_ring import binary_blocks
from code28_core import LIFT, MOTHER_K, MOTHER_N, OMITTED, SYMBOLS, SYSTEMATIC, coordinates, packed_check_columns, repair_one
from code28_receiver_core import calibration_consensus, corrected_grid, crossfit_gain, evaluate_grid
from educational_fec_demo import encode_block, word_bits
from quotient_lift_core import pascal_transform
from validate_code21 import HEADERS, SOURCE, digest
from validate_code28 import fit, labels_to_bits, predict
from validate_code37 import constellation_for

BASE = 'runs/v017-code28-hard/artifacts/experiments/E019/results'
QUOTIENT = 'runs/v018-code28-weighted-bank/artifacts/experiments/E020/results/code28_soft_quotient.npz'
CODE28_AUDIT = 'runs/v017-code28-verified/artifacts/experiments/E019/results/code28_boundaries.json'
SEED = 2026112021


def collect(iq, sss, records, points):
    prefix = word_bits(0x3048)[16:]
    sign = 1 - 2*encode_block(np.r_[prefix, np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    frames = np.array([r['frame'] for r in records])
    variants = {k: np.empty((len(records), SYMBOLS), dtype=np.complex64)
                for k in ['original', 'regrid', 'crossfit_only', 'regrid_crossfit']}
    receiver = []
    for frame in np.unique(frames):
        is_frame = frames == frame
        starts = np.array([r['start_resource'] for r in records if r['frame'] == frame])
        positions = starts[:, None] + np.arange(SYMBOLS)
        slot = positions.ravel()//1004
        carrier = positions.ravel()%1004
        old, a = evaluate_grid(fft_grid(iq, int(frame)), sss)
        regrid, b = evaluate_grid(corrected_grid(iq, int(frame), a['scale_delta']), sss)
        entry = {'frame': int(frame), 'before': a, 'after': b, 'decision_mse': {}}
        for key, folded in [('original', old), ('regrid', regrid)]:
            payload = folded.ravel()
            flat = (payload/reference_gain(payload[:48], sign))[positions].ravel()
            cross = crossfit_gain(crossfit_gain(flat, slot, carrier, points), carrier, slot, points)
            for name, z in [(key, flat), ('crossfit_only' if key=='original' else 'regrid_crossfit', cross)]:
                variants[name][is_frame] = z.reshape(-1, SYMBOLS)
                entry['decision_mse'][name] = float((abs(z[:, None]-points)**2).min(axis=-1).mean())
        receiver.append(entry)
        print('receiver', int(frame), a['scale_delta'], b['scale_delta'], entry['decision_mse'], flush=True)
    return variants, receiver, frames


def demap(z, points):
    distances = abs(z[..., None]-points)**2
    labels = distances.argmin(axis=-1)
    nearest = np.partition(distances, 1, axis=-1)[..., :2]
    margin = (nearest[..., 1]-nearest[..., 0]).min(axis=1)
    return distances, labels, margin


def main():
    require_run_context()
    out = experiment_output('E021')
    out.mkdir(parents=True, exist_ok=True)
    metadata = BASE + '/code28.json'
    observations = BASE + '/code28_observations.npz'
    input_hashes = {r:digest(ROOT/r) for r in [SOURCE, HEADERS, metadata, observations, QUOTIENT, CODE28_AUDIT]}
    source = json.loads((ROOT/metadata).read_text(encoding='utf-8'))
    audit = json.loads((ROOT/CODE28_AUDIT).read_text(encoding='utf-8'))
    if (input_hashes[SOURCE] != source['input_sha256'][SOURCE]
            or input_hashes[HEADERS] != source['input_sha256'][HEADERS]
            or input_hashes[metadata] != audit['baseline_result_sha256']
            or input_hashes[observations] != audit['input_npz_sha256']):
        raise RuntimeError('Archived evidence changed')
    sss_path = legacy_artifact('reference/qin/sssVec.mat')
    sss = loadmat(sss_path)['sssVecFull'].ravel()[LOADED % 1024]
    input_hashes['references/qin/sssVec.mat'] = digest(sss_path)
    iq = np.load(ROOT/SOURCE, mmap_mode='r', allow_pickle=False)
    records = source['words']
    points = constellation_for(2, False, True)
    variants, rx, frames = collect(iq, sss, records, points)
    with np.load(ROOT/observations, allow_pickle=False) as prior:
        groups = [prior['published_group'], prior['random_group']]
        original = prior['hard_bits']
    result = {'schema':'starlink-code28-full-reproduction/v1', 'input_sha256':input_hashes, 'receiver':rx,
              'words':records, 'branches':{}, 'fit_seed':SEED}
    arrays = {'original_hard_bits':original, 'published_training':groups[0], 'random_training':groups[1]}
    models = []
    for name, samples in variants.items():
        distances, labels, margin = demap(samples, points)
        bits = labels_to_bits(labels)
        if name=='original' and not np.array_equal(bits,original):
            raise RuntimeError('Original branch does not reproduce archived hard bits')
        u, v, mask = coordinates(bits)
        rows = []
        for i, group in enumerate(groups):
            high, row = fit(u, v, margin, group, frames)
            if high is not None:
                row['top_margin_raw_exact_words'] = int(np.count_nonzero(~(predict(u,high)^v).any(axis=1)))
            if name == 'regrid_crossfit':
                p, detail = calibration_consensus(u[group], v[group], LIFT, SEED+i)
                row['training_consensus'] = detail
                models.append(p)
            rows.append(row)
        result['branches'][name] = rows
        print(name, [(row['unit_pivots'],row['inconsistent_block_equations']) for row in rows], flush=True)
    primary = variants['regrid_crossfit']
    distances, labels, _ = demap(primary, points)
    hard = labels_to_bits(labels)
    u, v, mask = coordinates(hard)
    arrays.update(primary_symbols=primary, primary_hard_bits=hard, coordinate_mask=mask)
    result['full_matrix_agreement'] = bool(models[0] is not None and models[0]==models[1])
    if result['full_matrix_agreement']:
        p = models[0]
        seeds = binary_blocks(p,LIFT)
        full = seeds.reshape(9,7,LIFT)
        projection = pascal_transform(full)[...,:2]
        with np.load(ROOT/QUOTIENT, allow_pickle=False) as earlier:
            reference = earlier['weighted_case_0_group_0_coefficients'][...,:2]
        initial = predict(u,p)^v
        corrected = hard.copy()
        columns = packed_check_columns(p,LIFT,OMITTED,2)
        for i, row in enumerate(records):
            change, status = repair_one(labels[i], distances[i], initial[i], columns, mask, 256)
            corrected[i,change] ^= 1
            row.update(status, initial_failed_checks=int(initial[i].sum()), changed_positions=change)
        u2,v2,_ = coordinates(corrected)
        final = predict(u2,p)^v2
        symbols = mask.reshape(SYMBOLS,3).all(axis=1)
        for i,row in enumerate(records):
            row['final_failed_checks'] = int(final[i].sum())
            if row['status']=='resolved_in_search' and row['final_failed_checks']:
                raise RuntimeError('Resolved core did not re-encode')
            y = primary[i,symbols]
            alpha = corrected[i].reshape(SYMBOLS,3)[symbols] @ np.array([4,2,1],dtype=np.uint8)
            row['core_nmse'] = float(np.sum(abs(y-points[alpha])**2)/np.sum(abs(y)**2))
        for group,row in zip(groups,result['branches']['regrid_crossfit'],strict=True):
            row['reserved_words'] = int((~group).sum())
            row['reserved_initial_exact'] = int(np.count_nonzero(~group & ~initial.any(axis=1)))
            row['reserved_after_search_exact'] = int(np.count_nonzero(~group & ~final.any(axis=1)))
        raw_valid = np.flatnonzero(~initial.any(axis=1))
        first = int(raw_valid[0])
        prefix_columns = packed_check_columns(p[:1],LIFT,0,0)[:OMITTED]
        dim = MOTHER_N-MOTHER_K
        matrix = np.unpackbits(np.frombuffer(b''.join(x.to_bytes(dim//8,'little') for x in prefix_columns),
                                            dtype=np.uint8).reshape(OMITTED,dim//8),axis=1,bitorder='little').T
        completion = recover_generator(matrix, initial[first,:,None])
        completion_bits = completion.pop('generator')
        result.update(initial_exact_words=len(raw_valid), final_exact_words=int(np.count_nonzero(~final.any(axis=1))),
                      changed_core_bits=int(np.count_nonzero(hard!=corrected)), code_core_symbols=int(symbols.sum()),
                      generator_seed_sha256=hashlib.sha256(seeds.tobytes()).hexdigest(),
                      earlier_quotient_exact=bool(np.array_equal(reference,projection)),
                      earlier_quotient_matching_coefficients=int(np.count_nonzero(reference==projection)),
                      raw_prefix_completion={'word_index':first,**completion,
                                             'ones':None if completion_bits is None else int(completion_bits.sum())})
        arrays.update(generator_seed_bits=seeds, final_bits=corrected, initial_syndromes=initial,final_syndromes=final)
        print('Exact words:',result['initial_exact_words'],'->',result['final_exact_words'],'/',len(records),flush=True)
    (out/'code28_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'code28_reproduction.npz',**arrays)
    if not result['full_matrix_agreement']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
