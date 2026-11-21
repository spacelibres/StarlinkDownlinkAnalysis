"""Test held-row common phase between disjoint upper-edge LEO pilots."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_leo_pilot_archive import ARCHIVE, check, read, sha256


SEED = 20261121106
CONTROL_COUNT = 199
DATASETS = ('DS7', 'DS8', 'DS9')


def pilot_projections(matrix, times, frequency):
    if matrix.shape != (150, 8) or times.shape != (150,) or not np.isfinite(matrix).all():
        raise ValueError('Expected one finite 150-by-8 archive matrix and its time grid')
    corrected = matrix * np.exp(-2j * np.pi * frequency * times[:, None])
    gains = corrected[::2].mean(axis=0)
    return tuple(corrected[1::2, offset::2] @ gains[offset::2].conj() for offset in (0, 1))


def phase_curves(first, second, invert=False):
    """Signed alignment gain over the constant-phase baseline at every cyclic shift."""
    curves = []
    for predictor, target in ((first, second), (second, first)):
        amplitude = np.abs(predictor)
        weights = np.abs(target).sum()
        if weights == 0 or not np.any(amplitude):
            raise ValueError('Zero-energy pilot projection')
        unit = np.divide(predictor, amplitude, out=np.zeros_like(predictor), where=amplitude > 0)
        if invert:
            unit = unit.conj()
        baseline = target.real.sum() / weights
        alignment = np.fft.ifft(np.fft.fft(target) * np.fft.fft(unit).conj()).real / weights
        curves.append(alignment - baseline)
    return np.stack(curves)


def phase_gain(first, second, shift=0, invert=False):
    return phase_curves(first, second, invert=invert).mean(axis=0)[shift]


def audit(ledger):
    rng = np.random.default_rng(SEED)
    windows = []
    inputs = {}
    session_shifts = {}
    for dataset in DATASETS:
        paths = {name: ARCHIVE / dataset / name for name in ('result.json', 'spec.json', 'matrices.npz')}
        hashes = {name: check(path, ledger[path.relative_to(EXTERNAL_LEO_ROOT).as_posix()])
                  for name, path in paths.items()}
        result = read(paths['result.json'])
        spec = read(paths['spec.json'])
        if result['matrices_sha256'] != hashes['matrices.npz'] or result['spec_sha256'] != hashes['spec.json']:
            raise ValueError(f'Matrix or spec mismatch for {dataset}')
        inputs[dataset] = hashes
        with np.load(paths['matrices.npz'], allow_pickle=False) as archive:
            times = archive['times_s']
            if not np.all(np.diff(times) > 0):
                raise ValueError(f'Nonmonotone pilot time grid for {dataset}')
            session = spec['session_id']
            if session not in session_shifts:
                session_shifts[session] = rng.integers(1, len(times[1::2]), size=CONTROL_COUNT)
            shifts = session_shifts[session]
            for window in result['windows']:
                frames = []
                curves = []
                for frame in window['frames']:
                    frequency = next(method['fit']['frequency_hz'] for method in frame['methods']
                                     if method['method'] == 'ordinary')
                    first, second = pilot_projections(archive[frame['matrix_key']], times, frequency)
                    transfer = phase_curves(first, second)
                    curves.append(transfer.mean(axis=0))
                    frames.append({'matrix': frame['matrix_key'], 'frequency_hz': frequency,
                                   'directional_gains': transfer[:, 0].tolist(),
                                   'inverse_gain': float(phase_gain(first, second, invert=True))})
                mean_curve = np.mean(curves, axis=0)
                windows.append({'dataset': dataset, 'session': session, 'visit': window['visit_index'],
                                'receiver': window['receiver_id'], 'frames': frames,
                                'matched_gain': float(mean_curve[0]), 'one_row_shift_gain': float(mean_curve[1]),
                                'mean_inverse_gain': float(np.mean([f['inverse_gain'] for f in frames])),
                                'control_shifts_rows': shifts.tolist(), 'control_gains': mean_curve[shifts].tolist()})
    return windows, inputs


def main():
    require_run_context()
    path = ARCHIVE / 'evidence-sha256.json'
    windows, inputs = audit(read(path))
    controls = np.mean([window['control_gains'] for window in windows], axis=0)
    matched = float(np.mean([window['matched_gain'] for window in windows]))
    result = {'schema': 'starlink-leo-upper-pilot-residuals/v1', 'seed': SEED,
              'source_ledger_sha256': sha256(path), 'inputs_sha256': inputs, 'windows': windows,
              'aggregate': {'matched_gain': matched, 'control_maximum': float(controls.max()),
                            'controls_at_or_above_match': int(np.count_nonzero(controls >= matched)),
                            'control_gains': controls.tolist()}}
    destination = experiment_output('E106')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'leo_upper_pilot_residuals.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for dataset in DATASETS:
        rows = [window for window in windows if window['dataset'] == dataset]
        gains = [window['matched_gain'] for window in rows]
        print(f'{dataset}: {sum(len(w["frames"]) for w in rows)} frames; window gains {gains}')
    print(f'Matched gain {matched:.6f}; {result["aggregate"]["controls_at_or_above_match"]}/{CONTROL_COUNT} controls >= match')


if __name__ == '__main__':
    main()
