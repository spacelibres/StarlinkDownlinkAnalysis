"""Use frozen raw-body paths to decode H2, with CRC samples held out of selection."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_ut_offset_constrained_h2 import coded_bits, constrained_block, cosine, evaluate, legal_prefixes, local_prefix_phase
from audit_ut_repeated_data import checked_artifact, read
from layout_core import prefix_fields


PREFIX_RESOURCES = 24
CRC_RESOURCES = (8 + 6) * 3


def header_candidates(layout):
    blocks, entries = layout
    result = []
    for fixed in legal_prefixes(blocks):
        decoded = prefix_fields(fixed[:16])
        if decoded['blocks'] != blocks or decoded['entry_count'] != len(entries):
            continue
        word = np.zeros(32 * blocks, dtype=np.uint8)
        word[:16] = fixed[:16]
        for i, (code, count) in enumerate(entries):
            start = 16 + 20 * i
            word[start:start + 8] = (code >> np.arange(8)) & 1
            word[start + 8:start + 20] = (count >> np.arange(12)) & 1
        result.append(word)
    return result


def information_score(soft, bits, stop=None):
    signs = 1 - 2 * coded_bits(bits).astype(float)
    end = len(signs) - CRC_RESOURCES if stop is None else stop
    use = slice(PREFIX_RESOURCES, end)
    return cosine(soft[use].real, signs[use])


def check_crc_tail(soft, header):
    count = len(header) // 32 * 114
    known = header[-32:].astype(np.int8)[None]
    known[:, -8:] = -1
    last, _ = constrained_block(soft[count - 114:count].real, known)
    observed = header.copy()
    observed[-8:] = last[0, -8:]
    evidence = evaluate(observed, soft)
    signs = 1 - 2 * coded_bits(observed).astype(float)
    tail_score = cosine(soft[count - CRC_RESOURCES:count].real, signs[-CRC_RESOURCES:])
    return {**evidence, 'crc_region_cosine': tail_score}


def rank_paths(soft, paths):
    if not paths:
        return {'candidate_count': 0, 'selected': None}
    limit = min(blocks for blocks, _ in paths) * 114 - CRC_RESOURCES
    candidates = [word for layout in paths for word in header_candidates(layout)]
    scores = np.array([information_score(soft, bits, limit) for bits in candidates])
    indices = np.argsort(-scores, kind='stable')
    best, runner_up = scores[indices[:2]]
    info = {'header_bits': candidates[indices[0]].tolist(), 'pre_crc_score': float(best),
            'second_score': float(runner_up), 'ranking_stop_resource_in_header': limit}
    if np.isclose(best, runner_up, rtol=1e-10, atol=1e-12):
        return {'candidate_count': len(candidates), 'ambiguous': True, 'selected': None}
    return {'candidate_count': len(candidates), 'ambiguous': False,
            'selected': {**info, **check_crc_tail(soft, candidates[indices[0]])}}


def load_soft(source, config, h2):
    main, other = {}, {}
    for fold, row in enumerate(config['folds']):
        main.update(zip(row['held_ticks'], source[f'fold_{fold}_held'], strict=True))
        other.update(zip(row['train_ticks'], source[f'fold_{fold}_train'], strict=True))
    main.update(zip(config['overlap_ticks'], source['fold_0_overlap'], strict=True))
    other.update(zip(config['overlap_ticks'], source['fold_1_overlap'], strict=True))
    return [{tick: local_prefix_phase(x[h2:h2 + 798]) for tick, x in collection.items()} for collection in (main, other)]


def crc_summary(results):
    return sum(result['selected'] is not None and result['selected']['crc_ok'] for result in results)


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    a, b = [read(artifact('v108-ut-offset-body-markers' + suffix, 'E110', 'ut_offset_body_markers.json'))
            for suffix in ('', '-confirm')]
    if a != b:
        raise ValueError('Raw-body marker runs disagree')
    run = 'v042-ut-offset-grid-header'
    config = read(artifact(run, 'E044', 'ut_offset_grid_pilots.json'))
    with np.load(artifact(run, 'E044', 'ut_offset_grid_soft.npz'), allow_pickle=False) as source:
        primary, alternate = load_soft(source, config, a['h2_start'])
    results = []
    by_tick = {row['tick']: row['body_paths'] for row in a['rows']}
    for row in a['rows']:
        tick = row['tick']
        results.append({'tick': tick, 'group': row['group'],
                        'primary': rank_paths(primary[tick], by_tick[tick]),
                        'other_pilot_fit': rank_paths(alternate[tick], by_tick[tick])})
    controls = []
    for group in ('eight', 'seven', 'overlap'):
        ticks = [r['tick'] for r in a['rows'] if r['group'] == group]
        for lag in range(1, len(ticks)):
            moved = np.roll(ticks, lag).tolist()
            trials = []
            for tick, wrong in zip(ticks, moved, strict=True):
                try_wrong = rank_paths(primary[tick], by_tick[wrong])
                winner = try_wrong['selected']
                if winner is not None:
                    trials.append({'tick': tick, 'body_from_tick': wrong, 'crc_ok': winner['crc_ok'],
                                   'words_hex': winner['words_hex'], 'pre_crc_score': winner['pre_crc_score']})
            controls.append({'group': group, 'lag': lag, 'body_tick_mapping': list(zip(ticks, moved, strict=True)),
                             'selected_trials': trials})
    result = {'schema': 'starlink-ut-offset-body-first/v1', 'rows': results, 'whole_tick_controls': controls,
              'crc_outside_ranking_resources': CRC_RESOURCES, 'inputs_sha256': {p.as_posix(): h for p, h in hashes.items()}}
    out = experiment_output('E110')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ut_offset_body_first.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('CRC primary and alternate', [crc_summary([r[k] for r in results]) for k in ('primary', 'other_pilot_fit')], flush=True)
    for row in results:
        pick = row['primary']['selected']
        if pick is not None:
            print(row['tick'], pick['crc_ok'], pick['words_hex'], pick['pre_crc_score'], flush=True)


if __name__ == '__main__':
    main()
