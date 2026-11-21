"""Check the new H2 candidate on the original body, after freezing its CRC."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, legacy_artifact, require_run_context
from audit_blind_pn_boundaries import SYSTEMATIC_ENDS, WINDOW
from audit_internal_allocations import KINDS, WIDTH
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_offset_constrained_h2 import coded_bits, local_prefix_phase
from audit_ut_offset_grid_pilots import PILOT, SMOOTH_BINS
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED
from educational_fec_demo import word_bits
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import capture_grid
from verify_ut_offset_h2_body import audit_marker, map_pn, matched_pn, predicted_markers


SEED = 20261121124
CONTROL_COUNT = 63
HALF_WIDTH = 800


def measure_word_family(data, starts, code, reference, modes, shifts, keep):
    points, labels = modes[KINDS[code]]
    lanes = labels.shape[1]
    complete, mixed = divmod(SYSTEMATIC_ENDS[code], lanes)
    next_cell = complete + int(mixed != 0)
    before = map_pn(reference[(complete-WINDOW)*lanes:complete*lanes].reshape(-1, lanes), points, labels)
    after = map_pn(reference[next_cell*lanes:(next_cell+WINDOW)*lanes].reshape(-1, lanes), points, labels)
    index = starts[None, :, None] + shifts[:, None, None] + np.arange(WINDOW)
    fit = matched_pn(data[index + complete-WINDOW], before)
    follow = matched_pn(data[index + next_cell], after)
    folds = []
    for qualified in (keep, ~keep):
        m0, m1 = fit[:, qualified].mean(axis=1), follow[:, qualified].mean(axis=1)
        contrast = m0-m1
        folds.append({'mean_before': float(m0[0]), 'mean_continuation': float(m1[0]),
                      'contrast': float(contrast[0]), 'wrong_resource_contrasts': contrast[1:].tolist(),
                      'wrong_contrasts_ge_actual': int(np.count_nonzero(contrast[1:] >= contrast[0]))})
    return {'code': code, 'folds': folds, 'before_by_word': fit[0].tolist(), 'continuation_by_word': follow[0].tolist()}


def apparent_prefix_effect(words, alternate, soft):
    true_code = np.concatenate([word_bits(int(x, 16)) for x in words])
    alternative = np.concatenate([word_bits(int(x, 16)) for x in alternate])
    x, y = [1-2*coded_bits(b).astype(float) for b in (true_code, alternative)]
    affected = np.flatnonzero(x != y)
    delta = (x-y)*soft[:len(x)].real
    return {'changed_encoded_resources': affected.tolist(),
            'common_prefix_region_margin': float(delta[24:48].sum()),
            'convolution_memory_after_prefix_margin': float(delta[48:len(x)-42].sum()),
            'crc_region_margin_between_flag_variants': float(delta[len(x)-42:].sum())}


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    a, b = [doc('v117-ut-weak-two-allocations'+s, 'E119', 'ut_weak_tail_two_allocations.json') for s in ('', '-confirm')]
    if a != b:
        raise ValueError('Frozen two-allocation searches disagree')
    capture = doc('v022-ut-replay', 'E014', 'ut_transfer.json')
    config = doc('v042-ut-offset-grid-header', 'E044', 'ut_offset_grid_pilots.json')
    q = doc('v041-ut-dual-grid', 'E043', 'ut_weak_frames.json')
    sync = {r['tick']: r['sync'] for r in q['offset_grid']['records'] if 'sync' in r}
    h2 = doc('v116-ut-weak-tail-paths-markers', 'E118', 'ut_weak_tail_paths.json')['h2_start']
    model = doc('v027-qam-pn15-final', 'E029', 'qam_pn15_audit.json')['fit']['winner']
    ref = equality_predictions(np.arange(16384), pn_bits(), model)
    with np.load(artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz'), allow_pickle=False) as z:
        qam16 = z['constellation_points'], z['frozen_gray_mapping']
    mapping = doc('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')['cross32_mapping']
    qam32 = (np.array([complex(item['i'], item['q']) for item in mapping])/np.sqrt(20),
             labels_as_bits(np.array([item['label'] for item in mapping]), 5))
    modes = {'16QAM': qam16, 'cross32': qam32}
    sss_file = legacy_artifact('reference/qin/sssVec.mat')
    digest = sha256(sss_file)
    if digest != capture['sss_reference_sha256']:
        raise ValueError('SSS fingerprint changed')
    hashes[PROJECT_ROOT/'references/qin/sssVec.mat'] = digest
    sss = loadmat(sss_file)['sssVecFull'].ravel()[LOADED % 1024]
    raw = PROJECT_ROOT/SOURCE
    iq = load_samples(raw, capture)
    hashes[raw] = capture['source_sha256']
    slots = np.arange(2, 302)
    rotors = np.column_stack([pn_reference(int(k), slots) for k in LOADED[~PILOT]])
    rng = np.random.default_rng(SEED)
    shuffles = np.stack([rng.permutation(WINDOW//16) for _ in range(CONTROL_COUNT)])
    allowed = np.r_[np.arange(-HALF_WIDTH, 0), np.arange(1, HALF_WIDTH+1)]
    offsets = np.r_[0, rng.choice(allowed, CONTROL_COUNT, replace=False)]
    results = []
    for row in a['rows']:
        tick = row['tick']
        choices = row['trials'][0]['methods']
        joint = choices['joint']
        if not joint['common_crc_valid']:
            continue
        header = joint['primary']['selected']
        prefix = choices['prefix_first']['primary']['selected']
        original = choices['joint']['common_crc_valid']
        other = prefix['words_hex'] if prefix else original
        descriptor = header['layout']['entries']
        markers = predicted_markers(h2+header['coding_length'], descriptor)
        first, last = markers
        starts = first['allocation_start']+WIDTH[first['mcs']]*np.arange(first['count'])
        groups = np.arange(len(starts))//5
        selected = rng.permutation(np.unique(groups))[:len(np.unique(groups))//2]
        group_mask = np.isin(groups, selected)
        evidence = []
        for fold in config['folds']:
            timing = fold['timing']
            pk = sync[tick]
            grid, slots_present = capture_grid(iq, pk['epoch_480_samples'], timing['sampling_scale'],
                                                pk['cfo_hz']+timing['cfo_residual_hz'])
            channel = np.concatenate([gaussian_filter1d(x, SMOOTH_BINS) for x in np.split(grid[0]/sss, 2)])
            data = (grid[1:, ~PILOT]*rotors[:len(slots_present)-1].conj()*channel[~PILOT].conj()).ravel()
            z = local_prefix_phase(data[h2:])
            data = np.concatenate((data[:h2], z))
            checks = [measure_word_family(data, starts, code, ref, modes, offsets, group_mask)
                      for code in (first['mcs'], 66)]
            finish = audit_marker(data, last, ref, modes[KINDS[last['mcs']]], shuffles, True)
            evidence.append({'prefix_comparison': apparent_prefix_effect(original, other, z), 'first_allocation': checks,
                             'terminal_predicted_stop': finish})
        results.append({'tick': tick, 'words_hex': original, 'prefix_first_words_hex': other, 'markers': markers,
                        'first_allocation_training_groups': selected.tolist(), 'evidence': evidence})
        print(tick, 'prefix and body', [(e['prefix_comparison'],
              [(f['contrast'], f['wrong_contrasts_ge_actual']) for f in e['first_allocation'][0]['folds']],
              [e['terminal_predicted_stop'][key] for key in ('predicted_score','rank','peak_shift','continuation_score')])
              for e in evidence], flush=True)
    result = {'schema': 'starlink-two-tail-allocation-body/v1', 'seed': SEED, 'control_resource_shifts': offsets.tolist(),
              'input_sha256': {p.as_posix(): h for p, h in hashes.items()}, 'rows': results}
    out = experiment_output('E119')
    out.mkdir(parents=True, exist_ok=True)
    (out/'ut_weak_tail_two_body.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n',
                                                encoding='utf-8')


if __name__ == '__main__':
    main()
