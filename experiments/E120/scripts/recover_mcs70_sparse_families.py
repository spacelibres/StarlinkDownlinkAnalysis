"""Search distinct-block MCS 70 check families and test group-held transfer."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import collect, equality_predictions
from audit_sparse_qc_orbits import orbit_positions, summarize
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import fit_taps, held_only, packed_arrays, project_checks, totals
from search_mcs66_early_projections import PAIR_BITS, PAIR_ID_BITS, matching_pairs, signatures_for
from sparse_code5_core import binary_rank


SEED = 20261121125
LIFT = 512
WINDOW = 16
ROTATIONS = 8
REGISTER_WORDS = 4
MIN_VISIBLE = .98
MIN_RECUR = 2
MIN_SIGN_SUPPORT = 128
MAX_FULL_PROFILES = 32


def nominate_shapes(bits, mask, lift):
    coords, signatures, _ = signatures_for(bits, mask, 3)
    block, phase = np.divmod(coords, lift)
    groups = [np.flatnonzero(block == b) for b in range(bits.shape[1]//lift)]
    anchors = {int(block[i]): i for i in np.flatnonzero(phase == 0)}
    hits = set()
    for i, j, k in combinations(range(len(groups)), 3):
        if i not in anchors:
            continue
        anchor = anchors[i]
        left = groups[j]
        right = groups[k]
        keys = signatures[left]
        order = np.argsort(keys, kind='stable')
        target = signatures[anchor] ^ signatures[right]
        low = np.searchsorted(keys[order], target, side='left')
        high = np.searchsorted(keys[order], target, side='right')
        for r in np.flatnonzero(high > low):
            for l in left[order[low[r]:high[r]]]:
                hits.add(tuple(map(int, coords[[anchor, l, right[r]]])))

    tables = {}
    for i, j, k, l in combinations(range(len(groups)), 4):
        if i not in anchors:
            continue
        if (j, k) not in tables:
            u, v = np.meshgrid(groups[j], groups[k], indexing='ij')
            values = signatures[u] ^ signatures[v]
            packed = ((values << PAIR_BITS) | (u.astype(np.uint64) << PAIR_ID_BITS) | v.astype(np.uint64)).ravel()
            packed.sort()
            tables[j, k] = packed, packed >> PAIR_BITS
        pairs, keys = tables[j, k]
        anchor = anchors[i]
        target = signatures[anchor] ^ signatures[groups[l]]
        for t, b, c in matching_pairs(pairs, keys, target, np.uint64(0)):
            hits.add(tuple(map(int, coords[[anchor, b, c, groups[l][t]]])))
    return hits


def calibration_rows(meta, mask, edge):
    best = {}
    for index, row in enumerate(meta):
        visible = mask[index, edge:].mean()
        if not row['calibration'] or row['bound'] > LIFT or visible < MIN_VISIBLE:
            continue
        score = row['pn_errors']/max(row['pn_seen'], 1) - .1*visible
        group = row['group']
        if group not in best or score < best[group][0]:
            best[group] = score, index
    return np.array([best[group][1] for group in sorted(best)], dtype=int)


def local_search(bits, mask, rows, rng, lift=LIFT):
    n = bits.shape[1]
    if len(rows) < REGISTER_WORDS:
        return Counter(), []
    frequency = Counter()
    visits = []
    axes = np.arange(n)
    for start in range(0, lift, WINDOW):
        pick = rng.choice(rows, REGISTER_WORDS, replace=False)
        shifts = rng.choice(np.arange(start, start + WINDOW), ROTATIONS, replace=False)
        positions = axes//lift*lift + (axes+shifts[:, None]) % lift
        ref = bits[pick[0], positions]
        ref_seen = mask[pick[0], positions]
        rest = pick[1:, None, None]
        values = bits[rest, positions] ^ ref
        visible = mask[rest, positions] & ref_seen
        supported = nominate_shapes(values.reshape(-1, n), visible.reshape(-1, n), lift)
        frequency.update(sorted(supported))
        visits.append({'start': start, 'selected_rows': pick.tolist(), 'rotations': shifts.tolist(),
                       'three_term_nominations': sum(len(p)==3 for p in supported),
                       'four_term_nominations': sum(len(p)==4 for p in supported)})
    return frequency, visits


def signed_constants(bits, seen, groups):
    votes = np.zeros(bits.shape[1])
    coverage = np.zeros(bits.shape[1], dtype=int)
    for group in np.unique(groups):
        use = groups == group
        n = seen[use].sum(axis=0)
        k = (bits[use] & seen[use]).sum(axis=0)
        votes += (2*k-n)/np.maximum(n, 1)
        coverage += n > 0
    return votes > 0, (coverage >= 4) & (abs(votes) >= .75*coverage)


def score_syndrome(bits, masks, groups, support):
    syndrome, visible = project_checks(bits, masks, support)
    return totals(syndrome, visible, groups)


def candidate_rank(discovery, bits, mask, meta, edge):
    training = np.array([r['calibration'] and r['bound']<=LIFT for r in meta])
    groups = np.array([r['group'] for r in meta])
    result = []
    for support, count in discovery.items():
        if count < MIN_RECUR:
            continue
        parity = edge+orbit_positions([divmod(k, LIFT) for k in support], LIFT)
        value, seen = project_checks(bits[training], mask[training], parity)
        ref, clean = signed_constants(value, seen, groups[training])
        result.append((support, count, parity, ref, clean))
    return sorted(result, key=lambda item: (-int(item[-1].sum()), -item[1], item[0]))


def evaluate_candidate(candidate, bits, mask, meta, pn, non70, non66, raw70, edge):
    support, count, parity, ref, clean = candidate
    n = len(meta)
    groups = np.array([r['group'] for r in meta])
    training = np.array([r['calibration'] for r in meta])
    bounds = np.array([r['bound'] for r in meta])
    prefix = np.arange(edge) < bounds[:, None]
    active = (bits[:, :edge] ^ pn) & mask[:, :edge] & prefix
    par_value, par_seen = project_checks(bits, mask, parity)
    taps, trace = fit_taps(par_value ^ ref, par_seen & clean, active.reshape(n, -1, LIFT), bounds, groups, training)
    padded = ~training & (bounds <= LIFT)
    pad = summarize(par_value[padded], par_seen[padded], ref, clean, groups[padded])
    wrong = summarize(par_value[padded], par_seen[padded], np.roll(ref, 1),
                      clean & np.roll(clean, 1), groups[padded])
    row = {'parity_shape': [divmod(p, LIFT) for p in support], 'systematic_shape': taps,
           'recurrences': count, 'conditional_signs_covered': int(clean.sum()), 'held_conditional': pad,
           'held_shifted_conditional': wrong, 'systematic_fit': trace}
    if not taps:
        return row, None
    sys = orbit_positions(taps, LIFT)
    checks = np.column_stack((sys, parity))
    pn_replay = np.bitwise_xor.reduce(pn[sys], axis=1)
    shifted = sys//LIFT*LIFT + (sys+1) % LIFT
    score = score_syndrome(*non70, checks)
    control = score_syndrome(*non70, np.column_stack((shifted, parity)))
    other = score_syndrome(*non66, checks)
    direct = score_syndrome(*raw70, checks)
    row.update(pn_constants_recovered=int(np.count_nonzero(clean & (ref == pn_replay))),
               held_nonterminal_mcs70=score, held_wrong_systematic=control, held_mcs66=other,
               raw_iq_mcs70=direct)
    passed = (row['pn_constants_recovered'] >= .99*clean.sum() and score['seen'] > 10_000
              and score['group_accuracy'] >= .70 and direct['group_accuracy'] >= .70
              and max(control['group_accuracy'], other['group_accuracy']) <= .60)
    row['passed'] = bool(passed)
    return row, checks if passed else None


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    def doc(run, exp, name):
        return read(artifact(run, exp, name))

    meta, other = [doc('v118-mcs70-terminal'+s, 'E120', 'mcs70_terminal_words.json') for s in ('', '-confirm')]
    p0, p1 = [artifact('v118-mcs70-terminal'+s, 'E120', 'mcs70_terminal_words.npz') for s in ('', '-confirm')]
    if meta != other or hashes[p0] != hashes[p1]:
        raise ValueError('Terminal extraction replicas disagree')
    with np.load(p0, allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ('bits', 'mask'))
    edge, rows = meta['systematic_end'], meta['rows']
    pn = equality_predictions(np.arange(edge), pn_bits(), meta['pn_model']).astype(bool)
    fit_rows = calibration_rows(rows, mask, edge)
    discovery, visits = local_search(bits[:, edge:], mask[:, edge:], fit_rows, np.random.default_rng(SEED))
    candidates = candidate_rank(discovery, bits, mask, rows, edge)
    print('MCS 70 search', len(fit_rows), 'groups', len(discovery), 'shapes', len(candidates), 'recurring', flush=True)
    archive = doc('v080-gain-padded-words', 'E082', 'gain_padded_words.json')['cohorts']
    with np.load(artifact('v080-gain-padded-words', 'E082', 'gain_padded_words.npz'), allow_pickle=False) as z:
        non70, non66 = [held_only(*packed_arrays(z, (f'nonterminal_{code}_bits', f'nonterminal_{code}_seen')),
                                  archive[f'nonterminal_{code}']) for code in (70, 66)]
    path = artifact('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_symbols.npz')
    raw, _, _ = collect(path, doc('v026-ut-qam-extract-repeat', 'E028', 'ut_qam_extraction.json'))
    fine, _, _, raw_rows = raw[70]
    raw70 = fine.astype(bool), np.ones(fine.shape, bool), np.array([r['frame'] for r in raw_rows])
    profiles, supports = [], []
    for candidate in [r for r in candidates if r[-1].sum() >= MIN_SIGN_SUPPORT][:MAX_FULL_PROFILES]:
        row, matrix = evaluate_candidate(candidate, bits, mask, rows, pn, non70, non66, raw70, edge)
        profiles.append(row)
        if matrix is not None:
            supports.append(matrix)

    c = doc('v081-mcs66-padding-sweep', 'E083', 'mcs66_padding_words.json')
    with np.load(artifact('v081-mcs66-padding-sweep', 'E083', 'mcs66_padding_words.npz'), allow_pickle=False) as z:
        b66, m66 = packed_arrays(z, ('bits', 'mask'))
    chosen66 = calibration_rows(c['rows'], m66, c['systematic_end'])
    control, visits66 = local_search(b66[:, c['systematic_end']:], m66[:, c['systematic_end']:],
                                      chosen66, np.random.default_rng(SEED))
    prior = doc('v081-mcs66-systematic-checks', 'E083', 'mcs66_systematic_checks.json')['results']
    known = []
    for row in prior:
        shape = tuple(sorted(b*LIFT+i for b, i in row['parity_shape']))
        known.append({'shape': shape, 'recurrences': control[shape]})
    print('MCS 66 positive orbit recurrence', known, flush=True)

    out = experiment_output('E120')
    out.mkdir(parents=True, exist_ok=True)
    rank = 0
    if supports:
        matrix = np.zeros((LIFT*len(supports), bits.shape[1]), dtype=np.uint8)
        for j, support in enumerate(supports):
            matrix[j*LIFT+np.arange(LIFT)[:, None], support] = 1
        rank = binary_rank(matrix)
        np.savez_compressed(out/'mcs70_sparse_checks.npz', check_matrix_packed=np.packbits(matrix, axis=1))
    output = {'schema': 'starlink-mcs70-sparse-checks/v1', 'seed': SEED,
              'input_sha256': {p.as_posix(): h for p, h in hashes.items()},
              'lift': LIFT, 'register_rows': fit_rows.tolist(), 'registration': visits,
              'recurrence_histogram': dict(sorted(Counter(discovery.values()).items())),
              'recurring_calibration_shapes': [
                  {'support': r[0], 'recurrences': r[1], 'affine_calibration_rotations': int(r[-1].sum())}
                  for r in candidates], 'profiles': profiles, 'verified_rank': rank,
              'mcs66_positive_control': {'register_rows': chosen66.tolist(), 'registration': visits66,
                                         'known_matches': known,
                                         'recurrences': dict(sorted(Counter(control.values()).items()))}}
    (out/'mcs70_sparse_families.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Verified full-row rank', rank, 'profiled candidate shapes', len(profiles), flush=True)


if __name__ == '__main__':
    main()
