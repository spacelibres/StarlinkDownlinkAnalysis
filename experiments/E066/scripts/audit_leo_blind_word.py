"""Learn a cyclic binary shape using LEO IQ without a wideband word input."""

import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_pairs_live import REPORT
from audit_leo_pilot_phase_transfer import BLOCK_SLOTS, FIRST_SLOT, geometry
from audit_leo_tcode_iq_prediction import (
    PERIOD, RESOURCES_PER_SLOT, WORD_SIZE, contracted_gram, held_reduction, visible_carriers, waveform_bank,
)
from audit_narrowband_cycle import code_indices, permute_lanes
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_reference


SEED = 2026112166
CONTROLS = 99
ITERATIONS = 12
FLIPS = np.tile(np.array(list(product((-1, 1), repeat=4))), (1, PERIOD))


def word_indices(q):
    return (np.arange(WORD_SIZE)[:, None, None]+q[None, :, None]
            + RESOURCES_PER_SLOT*np.arange(PERIOD)) % WORD_SIZE


def split_axis(projected):
    order = np.argsort(projected, kind='stable')
    span = np.cumsum(projected[order])[:-1]
    n = np.arange(1, len(projected))
    split = np.argmax((span-n*projected.mean())**2/(n*(len(projected)-n)))+1
    signs = np.ones(len(projected), dtype=int)
    signs[order[:split]] = -1
    return signs*signs[0]


def pool_word(values, indices):
    word = np.zeros(WORD_SIZE, dtype=int)
    for lane in range(4):
        lane_indices = indices[:, :, 0] % 4 == lane
        observed = values[lane_indices]
        position = indices[lane_indices]//4
        if not len(observed):
            raise ValueError('Training IQ does not geometrically cover all four lanes')
        remapped = np.empty_like(observed)
        np.put_along_axis(remapped, position, observed, axis=1)
        gram = remapped.real.T @ remapped.real + remapped.imag.T @ remapped.imag
        _, directions = np.linalg.eigh(gram)
        word[lane::4] = split_axis(directions[:, -1])
    return word


def alignment(values, coordinates, signs):
    waves = signs[coordinates].astype(float)
    waves -= waves.mean(axis=-1, keepdims=True)
    waves /= np.maximum(np.linalg.norm(waves, axis=-1, keepdims=True), 1e-30)
    similarity = np.sum(abs(np.einsum('fki,pki->fpk', values, waves))**2, axis=2)
    phase = similarity.argmax(axis=1)
    return phase, float(similarity[np.arange(len(phase)), phase].sum())


def blind_word(values, q):
    coordinates = word_indices(q)
    best = {'energy': float('-inf')}
    for seed in range(len(values)):
        word = pool_word(values[seed:seed+1], coordinates[:1])
        for iteration in range(ITERATIONS):
            phase, score = alignment(values, coordinates, word)
            if score > best['energy']:
                best = {'word': word, 'phase': phase, 'energy': score, 'seed': seed, 'iteration': iteration}
            candidate = pool_word(values, coordinates[phase])
            if np.array_equal(candidate, word):
                break
            word = candidate
    return best


def cyclic_distance(first, second):
    candidates = np.stack([np.roll(second, -phase) for phase in range(WORD_SIZE)])
    raw = (candidates != first).reshape(WORD_SIZE, PERIOD, 4).sum(axis=1)
    distance = np.minimum(raw, PERIOD-raw).sum(axis=1)
    phase = int(distance.argmin())
    return {'mismatches': int(distance[phase]), 'phase': phase,
            'tied_phases': int(np.count_nonzero(distance == distance[phase])),
            'lane_mismatches': np.minimum(raw[phase], PERIOD-raw[phase]).tolist(),
            'lane_flips': (raw[phase] > PERIOD//2).tolist()}


def best_equivalent_forms(anchor, word):
    rotated = np.stack([np.roll(word, -phase) for phase in range(WORD_SIZE)])
    candidates = rotated[:, None, :]*FLIPS
    distance = np.count_nonzero(candidates != anchor, axis=2)
    return candidates[distance == distance.min()]


def consensus_positions(words):
    anchor = np.asarray(words[0])
    consistent = np.ones(WORD_SIZE, dtype=bool)
    matches = []
    for word in words:
        candidates = best_equivalent_forms(anchor, word)
        matches.append(len(candidates))
        consistent &= np.all(candidates == anchor, axis=0)
    return {'canonical_signs': np.where(consistent, anchor, 0).tolist(),
            'retained_positions': np.flatnonzero(consistent).tolist(),
            'erased_positions': np.flatnonzero(~consistent).tolist(),
            'tied_best_alignments_by_arm': matches}


def profile_background(nuisance, expanded, samples, calibration):
    matrix = np.column_stack((samples, expanded.reshape(len(expanded), -1)))
    factor = np.linalg.lstsq(nuisance[calibration], matrix[calibration], rcond=1e-9)[0]
    residual = matrix-nuisance @ factor
    target = residual[:, 0]
    flat = residual[:, 1:]
    coefficients, _, rank, _ = np.linalg.lstsq(flat[calibration], target[calibration], rcond=1e-9)
    coeff = coefficients.reshape(expanded.shape[1:])
    coeff -= coeff.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(abs(samples[calibration])**2)))
    return {'features': flat.reshape(expanded.shape), 'target': target, 'training': calibration,
            'calibration_coefficients': coeff/scale, 'rank': int(rank)}


def frame_signal(iq, w, n, gate, train, frame_clock, templates, carriers, frequencies):
    rate, cfo = w['rate_hz'], w['archived_cfo_hz']
    origin = w['epoch_sample']+n*rate/templates.FRAME_RATE_HZ
    clock = frame_clock(iq, rate, origin, cfo, 'lower')
    slot_s, cp_s = templates.OFDM_SYMBOL_DURATION_S, templates.CYCLIC_PREFIX_DURATION_S
    slot, local = geometry(clock, origin, rate, slot_s, cp_s)
    guard = gate['guard_samples']
    keep = (slot >= FIRST_SLOT) & (local >= cp_s+guard/rate-1e-12) & (local <= slot_s-guard/rate+1e-12)
    slot, local = slot[keep], local[keep]
    samples = clock.values[keep]*np.exp(-2j*np.pi*clock.time[keep]*gate['actual_cfo_hz'])
    training = np.isin((slot-FIRST_SLOT)//BLOCK_SLOTS, train)
    if (training.sum(), (~training).sum()) != (gate['calibration_samples'], gate['held_samples']):
        raise ValueError('E064 sample groups changed')
    pn = np.column_stack([pn_reference(int(k), slot) for k in carriers])
    base = pn*np.exp(2j*np.pi*(local[:, None]-cp_s)*frequencies)/np.sqrt(8)
    classes = (slot-2) % PERIOD
    expanded = base[:, :, None]*(classes[:, None, None] == np.arange(PERIOD))
    pilots = templates.qin_edge_pilot_symbols('lower')
    waveform = clock.basis[keep]
    atoms = waveform/pilots[slot-2]
    before = np.vstack((np.zeros((1, 8)), pilots[:-1]))[slot-2]
    after = np.vstack((pilots[1:], np.zeros((1, 8))))[slot-2]
    nuisance = np.column_stack((waveform, base, atoms*before, atoms*after))
    return profile_background(nuisance, expanded, samples, training)


def profile_phases(data, waves):
    features, target, mask = (data[k] for k in ('features', 'target', 'training'))
    gram, rhs = contracted_gram(features[mask], target[mask], waves)
    gains = (np.linalg.pinv(gram, rcond=1e-9, hermitian=True) @ rhs[..., None])[..., 0]
    score = np.einsum('fpk,fpk->fp', gains.conj(), rhs).real/np.vdot(target[mask], target[mask]).real
    return score, gains


def held_scores(frame, banks):
    profiles = [profile_phases(row, bank) for row, bank in zip(frame, banks, strict=True)]
    phase = (profiles[0][0]+profiles[1][0]).argmax(axis=1)
    scores, energy = [], []
    index = np.arange(len(phase))
    for data, bank, (_, gains) in zip(frame, banks, profiles, strict=True):
        mask = ~data['training']
        y = data['target'][mask]
        scores.append(held_reduction(data['features'][mask], y, bank[index, phase], gains[index, phase]))
        energy.append(np.vdot(y, y).real)
    return phase, np.stack(scores), np.asarray(energy)


def pair_modes(pair, windows, split, gates, rng, frame_clock, templates):
    inputs, carrier_lists, frequencies, raw = [], [], [], []
    for label in ('first_input', 'second_input'):
        key = gates['duplicate_windows'].get(pair[label], pair[label])
        window = windows[key]
        folder, array = key.split('/')
        with np.load(EXTERNAL_LEO_ROOT/REPORT/folder/'selected_iq.npz', allow_pickle=False) as archive:
            iq = archive[array]
        if hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest() != window['iq_sha256']:
            raise ValueError('E064 IQ fingerprint changed')
        observed, hz = visible_carriers(window['rate_hz'], window['archived_cfo_hz'],
                                        templates.SUBCARRIER_SPACING_HZ, templates.qin_edge_pilot_indices('lower'))
        inputs.append(window)
        carrier_lists.append(observed)
        frequencies.append(hz)
        raw.append(iq)
    q = np.concatenate([code_indices(c) for c in carrier_lists])
    support = [i for i, raw_frame in enumerate(zip(*(w['frames'] for w in inputs), strict=True))
               if all(g['positive'] for arm in raw_frame for g in arm)]
    modes = []
    for guard_index in (0, 1):
        sampled = {}
        for n in support:
            sampled[n] = [frame_signal(iq, w, n, w['frames'][n][guard_index], gates['calibration_blocks'],
                                       frame_clock, templates, carriers, hz)
                          for iq, w, carriers, hz in zip(raw, inputs, carrier_lists, frequencies, strict=True)]
        for learning_frames, testing_frames in split:
            learn = [f for f in learning_frames if f in sampled]
            held = [f for f in testing_frames if f in sampled]
            coeff = np.stack([np.concatenate([r['calibration_coefficients'] for r in sampled[i]]) for i in learn])
            fit = blind_word(coeff, q)
            word = fit['word']
            bank = np.stack([word, *[permute_lanes(word, rng) for _ in range(CONTROLS)]])
            banks = [waveform_bank(bank, carriers) for carriers in carrier_lists]
            observations, sums, powers = [], np.zeros((2, len(bank))), np.zeros(2)
            for i in held:
                phases, reductions, energy = held_scores(sampled[i], banks)
                observations.append({'frame': i, 'phases': phases.tolist(),
                                     'held_raw_improvement': reductions.tolist(), 'held_raw_energy': energy.tolist()})
                sums += reductions
                powers += energy
            skill = sums/powers[:, None]
            modes.append({'guard': inputs[0]['frames'][0][guard_index]['guard_samples'],
                          'learning_frames': learn, 'held_frames': held, 'learned_word': word.tolist(),
                          'training_phases': fit['phase'].tolist(), 'training_energy': fit['energy'],
                          'initialization_frame': learn[fit['seed']], 'iteration': fit['iteration'],
                          'projected_ranks': [[r['rank'] for r in sampled[f]] for f in support],
                          'control_words': bank[1:].tolist(), 'held': observations,
                          'held_improvement_by_receiver': skill.tolist()})
            print(pair['pair_time_s'], modes[-1]['guard'], learn, 'held', skill[:, 0].round(5), flush=True)
    return {'time_s': pair['pair_time_s'], 'receiver_keys': [w['key'] for w in inputs],
            'carriers': [c.tolist() for c in carrier_lists], 'modes': modes}


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    inputs = {}
    def archive(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))
    original = archive('v054-leo-cfo-confirm', 'E056', 'leo_cfo_aperture.json')
    gates = archive('v062-leo-pilot-phase-global', 'E064', 'leo_pilot_phase_transfer.json')
    for rel, expected in original['external_inputs_sha256'].items():
        p = EXTERNAL_LEO_ROOT/rel
        inputs[p] = sha256(p)
        if inputs[p] != expected:
            raise ValueError(f'External source changed: {rel}')
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT/'src'), str(EXTERNAL_LEO_ROOT/'tools')]
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    rng = np.random.default_rng(SEED)
    groups = np.arange(14).reshape(7, 2)
    order = rng.permutation(len(groups))
    a, b = [np.sort(groups[index].ravel()).tolist() for index in (order[:3], order[3:])]
    split = ((a, b), (b, a))
    windows = {w['key']: w for w in gates['windows']}
    pairs = [pair_modes(pair, windows, split, gates, rng, FrameClock, templates) for pair in original['pairs']]
    for p in pairs:
        minimum = np.min([m['held_improvement_by_receiver'] for m in p['modes']], axis=(0, 1))
        p['minimum_skill'] = minimum.tolist()
        p['controls_ge_blind'] = int(np.count_nonzero(minimum[1:] >= minimum[0]))
        p['blind_consensus'] = consensus_positions([m['learned_word'] for m in p['modes']])
    result = {'schema': 'leo-blind-word/v2', 'seed': SEED, 'max_iterations': ITERATIONS,
              'training_block_numbers': gates['calibration_blocks'], 'held_block_numbers': gates['held_blocks'],
              'chronological_frame_group_count': len(groups), 'frame_group_permutation': order.tolist(),
              'inputs_sha256': {p.as_posix(): h for p, h in inputs.items()}, 'pairs': pairs}
    out = experiment_output('E066')
    out.mkdir(parents=True, exist_ok=True)
    (out/'leo_blind_word.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
