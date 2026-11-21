"""Refine the blind 60-sign LEO candidate on raw calibration samples only."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_blind_word import consensus_positions, frame_signal, profile_phases
from audit_leo_pairs_live import REPORT
from audit_leo_tcode_iq_prediction import held_reduction, visible_carriers, waveform_bank
from audit_narrowband_cycle import code_indices, permute_lanes
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read


SEED = 20261121108
CONTROL_COUNT = 99
OUTER_STEPS = 8
WORD_LENGTH = 60


def signed_design(features, indices, gains):
    design = np.zeros((len(features), WORD_LENGTH), complex)
    for carrier, positions in enumerate(indices):
        design[:, positions] += features[:, carrier, :] * gains[carrier]
    return design


def flip_descent(quadratic, linear, initial):
    signs = initial.copy()
    diagonal = np.diag(quadratic)
    drops = []
    for _ in range(len(signs)):
        improvements = 4 * (signs * (quadratic @ signs - linear) - diagonal)
        best = int(improvements.argmax())
        if improvements[best] <= 1e-12:
            break
        drops.append(float(improvements[best]))
        signs[best] *= -1
    return signs, drops


def best_phases(scores):
    return [np.flatnonzero(np.isclose(row, row.max(), rtol=1e-10, atol=1e-12)) for row in scores]


def calibrate(training, carriers, bank):
    waves = [waveform_bank(bank, indices) for indices in carriers]
    profiles = [[profile_phases(data, wave) for data, wave in zip(pair, waves, strict=True)] for pair in training]
    shape_score = 0.0
    phases = []
    gains = []
    for profiles_at_frame in profiles:
        scores = profiles_at_frame[0][0] + profiles_at_frame[1][0]
        phase = int(scores[0].argmax())
        shape_score += float(scores[0, phase])
        phases.append(phase)
        gains.append([item[1][0, phase] for item in profiles_at_frame])
    return shape_score, phases, gains


def profiled_flip(training, carriers, signs):
    words = np.broadcast_to(signs, (WORD_LENGTH, WORD_LENGTH)).copy()
    np.fill_diagonal(words, -signs)
    waves = [waveform_bank(words, bins) for bins in carriers]
    objectives = np.zeros(WORD_LENGTH)
    for pair in training:
        first, second = [profile_phases(data, wave)[0] for data, wave in zip(pair, waves, strict=True)]
        objectives += (first + second).max(axis=1)
    best = int(objectives.argmax())
    return words[best], float(objectives[best])


def fit_word(training, carriers, initial):
    coordinates = [(code_indices(bins)[:, None] + 1004 * np.arange(15)) % WORD_LENGTH for bins in carriers]
    signs = initial.copy()
    score, phases, gains = calibrate(training, carriers, signs[None])
    history = [{'score': score, 'flips': 0, 'update': 'initial'}]
    for _ in range(OUTER_STEPS):
        quadratic = np.zeros((WORD_LENGTH, WORD_LENGTH))
        linear = np.zeros(WORD_LENGTH)
        for pair, phase, gain_pair in zip(training, phases, gains, strict=True):
            for data, indices, gain in zip(pair, coordinates, gain_pair, strict=True):
                mask = data['training']
                y = data['target'][mask]
                matrix = signed_design(data['features'][mask], (indices + phase) % WORD_LENGTH, gain)
                power = np.vdot(y, y).real
                quadratic += (matrix.conj().T @ matrix).real / power
                linear += (matrix.conj().T @ y).real / power
        updated, drops = flip_descent(quadratic, linear, signs)
        if drops:
            method = 'fixed_phase_and_gain'
        else:
            updated, profiled_score = profiled_flip(training, carriers, signs)
            if profiled_score <= score + 1e-12:
                break
            drops = [profiled_score - score]
            method = 'fully_profiled_single_flip'
        if np.array_equal(updated, signs):
            break
        next_score, phases, gains = calibrate(training, carriers, updated[None])
        if next_score < score + sum(drops) - 1e-9:
            raise ValueError('Binary update failed to improve the profiled calibration objective')
        signs, score = updated, next_score
        history.append({'score': score, 'flips': len(drops), 'update': method})
    return signs, history


def test_frame(pair, waves):
    profiles = [profile_phases(data, wave) for data, wave in zip(pair, waves, strict=True)]
    joint = best_phases(profiles[0][0] + profiles[1][0])
    parts = []
    for rx, data in enumerate(pair):
        other = 1 - rx
        donor = best_phases(profiles[other][0])
        mask = ~data['training']
        y = data['target'][mask]
        selections = {}
        for label, phases in (('joint', joint), ('donor', donor)):
            owners = np.repeat(np.arange(len(phases)), [len(p) for p in phases])
            choice = np.concatenate(phases)
            reduction = held_reduction(data['features'][mask], y, waves[rx][owners, choice],
                                       profiles[rx][1][owners, choice])
            sizes = np.array([len(p) for p in phases])
            values = np.add.reduceat(reduction, np.r_[0, sizes.cumsum()[:-1]]) / sizes
            selections[label] = {'reduction': values.tolist(), 'phase_sets_for_words': [p.tolist() for p in phases[:2]]}
        parts.append({'held_energy': float(np.vdot(y, y).real), **selections})
    return parts


def audit_pair(pair, gates, templates, frame_clock, rng):
    gates_by_key = {row['key']: row for row in gates['windows']}
    windows = [gates_by_key[key] for key in pair['receiver_keys']]
    carriers = [np.asarray(c) for c in pair['carriers']]
    raw_arrays, frequencies = [], []
    for window, old_carriers in zip(windows, carriers, strict=True):
        folder, array = window['key'].split('/')
        with np.load(EXTERNAL_LEO_ROOT / REPORT / folder / 'selected_iq.npz', allow_pickle=False) as source:
            iq = source[array]
        if hashlib.sha256(iq.astype('<c8', copy=False).tobytes()).hexdigest() != window['iq_sha256']:
            raise ValueError(f"Stale raw IQ {window['key']}")
        raw_arrays.append(iq)
        actual, hz = visible_carriers(window['rate_hz'], window['archived_cfo_hz'],
                                      templates.SUBCARRIER_SPACING_HZ, templates.qin_edge_pilot_indices('lower'))
        if not np.array_equal(actual, old_carriers):
            raise ValueError('Nonpilot support differs from E066')
        frequencies.append(hz)
    by_guard = {}
    arms = []
    for arm in pair['modes']:
        guard = arm['guard']
        train, hold = arm['learning_frames'], arm['held_frames']
        if guard not in by_guard:
            guard_index = [row['guard_samples'] for row in windows[0]['frames'][0]].index(guard)
            by_guard[guard] = {
                frame: [frame_signal(iq, window, frame, window['frames'][frame][guard_index],
                                     gates['calibration_blocks'], frame_clock, templates, bins, hz)
                        for iq, window, bins, hz in zip(raw_arrays, windows, carriers, frequencies, strict=True)]
                for frame in sorted(train + hold)
            }
        local = by_guard[guard]
        initial = np.asarray(arm['learned_word'])
        refined, objective = fit_word([local[frame] for frame in train], carriers, initial)
        bank = np.stack([initial, refined, *[permute_lanes(refined, rng) for _ in range(CONTROL_COUNT)]])
        waves = [waveform_bank(bank, bins) for bins in carriers]
        held = [{'frame': frame, 'receivers': test_frame(local[frame], waves)} for frame in hold]
        scores = {}
        for label in ('joint', 'donor'):
            scores[label] = []
            for rx in range(2):
                rows = [sample['receivers'][rx] for sample in held]
                scores[label].append((np.sum([row[label]['reduction'] for row in rows], axis=0) /
                                      sum(row['held_energy'] for row in rows)).tolist())
        arms.append({'guard': guard, 'learning_frames': train, 'held_frames': hold,
                     'initial_word': initial.tolist(), 'refined_word': refined.tolist(), 'objective': objective,
                     'control_words': bank[2:].tolist(), 'held': held, 'scores': scores})
        print(f"{pair['time_s']:.3f}s {guard} {len(train)} frames, raw objective {objective[0]['score']:.5f} -> "
              f"{objective[-1]['score']:.5f}, held donor {np.asarray(scores['donor'])[:, :2].round(5).tolist()}",
              flush=True)
    return {'time_s': pair['time_s'], 'receiver_keys': pair['receiver_keys'], 'carriers': pair['carriers'],
            'initial_consensus': pair['blind_consensus'],
            'refined_consensus': consensus_positions([arm['refined_word'] for arm in arms]), 'arms': arms}


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    inputs = {}

    def artifact(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    archive = artifact('v064-leo-blind-consensus', 'E066', 'leo_blind_word.json')
    confirmed = artifact('v064-leo-blind-consensus-confirm', 'E066', 'leo_blind_word.json')
    if archive != confirmed:
        raise ValueError('Frozen E066 estimates disagree')
    original = artifact('v054-leo-cfo-confirm', 'E056', 'leo_cfo_aperture.json')
    gates = artifact('v062-leo-pilot-phase-global', 'E064', 'leo_pilot_phase_transfer.json')
    # Author FrameClock imports are read-only and receive precisely the E066 raw apertures.
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT / 'src'), str(EXTERNAL_LEO_ROOT / 'tools')]
    for relative, expected in original['external_inputs_sha256'].items():
        source = EXTERNAL_LEO_ROOT / relative
        inputs[source] = sha256(source)
        if inputs[source] != expected:
            raise ValueError(f'External source changed: {relative}')
    from dsx01_pilot_clock import FrameClock
    from leo.analysis.starlink import templates

    rng = np.random.default_rng(SEED)
    pairs = [audit_pair(pair, gates, templates, FrameClock, rng) for pair in archive['pairs']]
    result = {'schema': 'starlink-leo-profiled-word/v1', 'seed': SEED,
              'input_sha256': {path.as_posix(): value for path, value in inputs.items()},
              'calibration_blocks': gates['calibration_blocks'], 'held_blocks': gates['held_blocks'], 'pairs': pairs}
    out = experiment_output('E108')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'leo_profiled_word.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
