"""Predict held LEO raw samples with the independently frozen 60-position word."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_pairs_live import REPORT
from audit_leo_pilot_phase_transfer import BLOCK_SLOTS, FIRST_SLOT, geometry
from audit_narrowband_cycle import code_indices, permute_lanes
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_reference


SEED = 2026112165
CONTROL_WORDS = 99
PERIOD = 15
WORD_SIZE = 60
RESOURCES_PER_SLOT = 1004
MODELS = ('current_pilots', 'also_adjacent_pilot_spill')


def waveform_bank(words, carriers):
    q = code_indices(carriers)
    index = (np.arange(WORD_SIZE)[:, None, None] + q[None, :, None]
             + RESOURCES_PER_SLOT*np.arange(PERIOD)) % WORD_SIZE
    return words[:, index].astype(float)


def contracted_gram(features, signals, waves):
    _, count, classes = features.shape
    gram = (features.reshape(len(features), -1).conj().T @
            features.reshape(len(features), -1)).reshape(count, classes, count, classes)
    target = np.einsum('nki,n->ki', features.conj(), signals)
    rhs = np.einsum('fpki,ki->fpk', waves, target)
    result = np.empty(waves.shape[:2]+(count, count), dtype=complex)
    for i in range(count):
        for k in range(i, count):
            values = np.sum((waves[..., i, :] @ gram[i, :, k, :])*waves[..., k, :], axis=-1)
            result[..., i, k] = values
            result[..., k, i] = values.conj()
    return result, rhs


def held_reduction(features, samples, waves, gains):
    design = np.einsum('nki,fki->fnk', features, waves)
    predicted = np.einsum('fnk,fk->fn', design, gains)
    return np.sum(2*(samples.conj()*predicted).real-abs(predicted)**2, axis=1)


def score_families(nuisance, expanded, samples, calibration, waves):
    matrix = np.column_stack((samples, expanded.reshape(len(expanded), -1)))
    profile = np.linalg.lstsq(nuisance[calibration], matrix[calibration], rcond=1e-9)[0]
    residual = matrix - nuisance @ profile
    target = residual[:, 0]
    features = residual[:, 1:].reshape(expanded.shape)
    gram, rhs = contracted_gram(features[calibration], target[calibration], waves)
    gains = (np.linalg.pinv(gram, rcond=1e-9, hermitian=True) @ rhs[..., None])[..., 0]
    score = np.einsum('fpk,fpk->fp', gains.conj(), rhs).real
    best = score.argmax(axis=1)
    index = np.arange(len(waves))
    y = target[~calibration]
    held_features = features[~calibration]
    reduction = held_reduction(held_features, y, waves[index, best], gains[index, best])
    reference_phases = held_reduction(held_features, y, waves[0], gains[0])
    ties = np.count_nonzero(np.isclose(score[0], score[0, best[0]], rtol=1e-8, atol=1e-9))
    return {'phase': best, 'held_error_reduction': reduction, 'reference_tied_phases': int(ties),
            'held_background_error': float(np.vdot(y, y).real),
            'reference_phase_held_error_reduction': reference_phases}


def visible_carriers(rate, cfo, tone_hz, pilot_indices):
    all_nearby = np.arange(520, 545)
    frequencies = (all_nearby-np.mean(pilot_indices))*tone_hz
    band = (abs(frequencies+cfo) < .49*rate) & ~np.isin(all_nearby, pilot_indices)
    return all_nearby[band], frequencies[band]


def measure_frame(iq, rate, origin, cfo, edge, frequencies, carriers, waves, saved, train, frame_clock, templates):
    clock = frame_clock(iq, rate, origin, cfo, edge)
    slot_s, cp_s = templates.OFDM_SYMBOL_DURATION_S, templates.CYCLIC_PREFIX_DURATION_S
    slot, local = geometry(clock, origin, rate, slot_s, cp_s)
    block = (slot-FIRST_SLOT)//BLOCK_SLOTS
    reference = np.column_stack([pn_reference(int(k), slot) for k in carriers])
    atoms = reference*np.exp(2j*np.pi*(local[:, None]-cp_s)*frequencies)/np.sqrt(8)
    classes = (slot-2) % PERIOD
    expanded = atoms[:, :, None]*(classes[:, None, None] == np.arange(PERIOD))
    pilots = templates.qin_edge_pilot_symbols(edge)
    pilot_atoms = clock.basis/pilots[slot-2]
    prev = np.vstack((np.zeros((1, 8)), pilots[:-1]))[slot-2]
    after = np.vstack((pilots[1:], np.zeros((1, 8))))[slot-2]
    basic = np.column_stack((clock.basis, atoms))
    spill = np.column_stack((basic, pilot_atoms*prev, pilot_atoms*after))
    rows = []
    for gate in saved:
        guard = gate['guard_samples']
        keep = (block >= 0) & (local >= cp_s+guard/rate-1e-12) & (local <= slot_s-guard/rate+1e-12)
        cal = np.isin(block[keep], train)
        if (cal.sum(), (~cal).sum()) != (gate['calibration_samples'], gate['held_samples']):
            raise ValueError('Frame alignment does not match E064')
        residual_cfo = gate['actual_cfo_hz']
        samples = clock.values[keep]*np.exp(-2j*np.pi*clock.time[keep]*residual_cfo)
        for name, nuisance in zip(MODELS, (basic, spill), strict=True):
            result = score_families(nuisance[keep], expanded[keep], samples, cal, waves)
            rows.append({'guard_samples': guard, 'background': name,
                         'phase': result['phase'].tolist(), 'tied_reference_phases': result['reference_tied_phases'],
                         'held_background_error': result['held_background_error'],
                         'held_error_reduction': result['held_error_reduction'].tolist(),
                         'reference_phase_held_error_reduction':
                             result['reference_phase_held_error_reduction'].tolist()})
    return rows


def assess_pair(windows, aliases, pair):
    keys = [aliases.get(pair[label], pair[label]) for label in ('first_input', 'second_input')]
    first, second = ({r['frame']: r['models'] for r in windows[name]['frames']} for name in keys)
    common = sorted(first.keys() & second.keys())
    directions = []
    for donor, target, donor_key, target_key in ((first, second, *keys), (second, first, *keys[::-1])):
        paths = []
        for path in range(4):
            phases = np.array([donor[f][path]['phase'][0] for f in common])
            score = np.array([target[f][path]['reference_phase_held_error_reduction'] for f in common])
            energy = sum(target[f][path]['held_background_error'] for f in common)
            paired = [float(score[np.arange(len(common)), np.roll(phases, lag)].sum()/energy)
                      for lag in range(len(common))]
            paths.append({'donor_selected_phase': phases.tolist(), 'aligned_skill': paired[0],
                          'rolled_frame_skill': paired[1:],
                          'passes': paired[0] > max(0., max(paired[1:]))})
        directions.append({'donor': donor_key, 'target': target_key, 'paths': paths})
    return {'time_s': pair['pair_time_s'], 'frames': common, 'directions': directions}


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    inputs = {}
    def archived(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))
    old = archived('v054-leo-cfo-confirm', 'E056', 'leo_cfo_aperture.json')
    gates = archived('v062-leo-pilot-phase-global', 'E064', 'leo_pilot_phase_transfer.json')
    tail = archived('v023-binary-tail-final', 'E025', 'ut_binary_tail.json')
    for rel, fingerprint in old['external_inputs_sha256'].items():
        path = EXTERNAL_LEO_ROOT/rel
        inputs[path] = sha256(path)
        if inputs[path] != fingerprint:
            raise ValueError(f'External source differs from E056: {rel}')
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT/'src'), str(EXTERNAL_LEO_ROOT/'tools')]
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    word = np.asarray(tail['cycle_folds'][0]['signed_template'])
    rng = np.random.default_rng(SEED)
    words = np.stack([word, *[permute_lanes(word, rng) for _ in range(CONTROL_WORDS)]])
    train = gates['calibration_blocks']
    windows, window_minima = [], []
    for w in gates['windows']:
        key = w['key']
        folder, array = key.split('/')
        config = old['windows'][key]
        edge, rate, cfo = config['edge'], w['rate_hz'], w['archived_cfo_hz']
        if edge != 'lower':
            raise ValueError('Resource mapping currently expects lower edge')
        with np.load(EXTERNAL_LEO_ROOT/REPORT/folder/'selected_iq.npz', allow_pickle=False) as archive:
            iq = archive[array]
        if hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest() != w['iq_sha256']:
            raise ValueError('IQ does not match E064')
        pilots = templates.qin_edge_pilot_indices(edge)
        carriers, frequencies = visible_carriers(rate, cfo, templates.SUBCARRIER_SPACING_HZ, pilots)
        waves = waveform_bank(words, carriers)
        frames = []
        sums = np.zeros((4, len(words)))
        energies = np.zeros(4)
        for index, pair in enumerate(w['frames']):
            if not all(gate['positive'] for gate in pair):
                continue
            epoch = w['epoch_sample']+index*rate/templates.FRAME_RATE_HZ
            rows = measure_frame(iq, rate, epoch, cfo, edge, frequencies, carriers, waves, pair, train,
                                 FrameClock, templates)
            frames.append({'frame': index, 'models': rows})
            sums += np.array([r['held_error_reduction'] for r in rows])
            energies += np.array([r['held_background_error'] for r in rows])
        skill = sums/energies[:, None]
        window_minima.append(skill.min(axis=0))
        windows.append({'key': key, 'carriers': carriers.tolist(), 'frames': frames,
                        'per_path_held_error_improvement': skill.tolist()})
        print(key, carriers.tolist(), 'frames', len(frames), 'improvement', skill[:, 0].round(4), flush=True)
    null = np.max(np.asarray(window_minima)[:, 1:], axis=0)
    for w, scores in zip(windows, window_minima, strict=True):
        w['minimum_path_improvement'] = float(scores[0])
        w['joint_controls_ge_reference'] = int(np.count_nonzero(null >= scores[0]))
    keyed = {w['key']: w for w in windows}
    pairs = [assess_pair(keyed, gates['duplicate_windows'], pair) for pair in old['pairs']]
    result = {'schema': 'leo-tcode-iq-prediction/v2', 'seed': SEED, 'sequence_controls': words.tolist(),
              'inputs_sha256': {k.as_posix(): v for k, v in inputs.items()}, 'calibration_blocks': train,
              'held_blocks': gates['held_blocks'], 'window_max_of_control_minima': null.tolist(), 'windows': windows,
              'paired_receiver_phase_transfer': pairs}
    out = experiment_output('E065')
    out.mkdir(parents=True, exist_ok=True)
    (out/'leo_tcode_iq_prediction.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
