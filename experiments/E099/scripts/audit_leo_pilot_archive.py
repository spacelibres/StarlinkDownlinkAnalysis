"""Recompute held-symbol scores from the LEO DS7/DS8/DS9 pilot archives."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context


ARCHIVE = EXTERNAL_LEO_ROOT / 'reports/2026_09_28_pilot_split_transfer'
TOLERANCE = 1e-12


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return actual


def score(samples, times, frequency):
    kernel = np.exp(2j * np.pi * frequency * times)
    projection = kernel.conj() @ samples
    return float(np.vdot(projection, projection).real / (len(times) * np.vdot(samples, samples).real))


def audit_dataset(dataset, ledger):
    source = ARCHIVE / dataset
    paths = {name: source / name for name in ('result.json', 'spec.json', 'matrices.npz')}
    hashes = {name: check(path, ledger[path.relative_to(EXTERNAL_LEO_ROOT).as_posix()])
              for name, path in paths.items()}
    result = read(paths['result.json'])
    spec = read(paths['spec.json'])
    if result['matrices_sha256'] != hashes['matrices.npz'] or result['spec_sha256'] != hashes['spec.json']:
        raise ValueError(f'Inconsistent matrix or spec binding for {dataset}')
    check(EXTERNAL_LEO_ROOT / spec['baseline_path'], spec['baseline_sha256'])
    max_error = 0.0
    checks = 0
    windows = []
    with np.load(paths['matrices.npz'], allow_pickle=False) as archive:
        times = archive['times_s']
        if times.shape != (150,) or not np.all(np.diff(times) > 0):
            raise ValueError(f'Unexpected OFDM time grid in {dataset}')
        for window in result['windows']:
            gains = {'ordinary': [], 'robust': []}
            for frame in window['frames']:
                matrix = archive[frame['matrix_key']]
                if matrix.shape != (150, 8):
                    raise ValueError(f'Unexpected matrix in {dataset}/{frame["matrix_key"]}')
                held = matrix[1::2]
                phase = 1j ** np.random.default_rng(frame['seed']).integers(0, 4, len(held))
                scrambled = held * phase[:, None]
                baseline = score(held, times[1::2], 0.0)
                fits = [(0.0, frame['baseline_held_coherence'], frame['baseline_scrambled_coherence'])]
                for method in frame['methods']:
                    frequency = method['fit']['frequency_hz']
                    fits.append((frequency, method['held_coherence'], method['scrambled_coherence']))
                    gains[method['method']].append(score(held, times[1::2], frequency) - baseline)
                for frequency, real, control in fits:
                    for samples, expected in ((held, real), (scrambled, control)):
                        error = abs(score(samples, times[1::2], frequency) - expected)
                        if not error < TOLERANCE:
                            raise ValueError(f'{dataset}/{frame["matrix_key"]}: coherence error {error}')
                        max_error = max(max_error, error)
                        checks += 1
            windows.append({'visit': window['visit_index'], 'receiver': window['receiver_id'],
                            'frames': len(window['frames']), 'mean_held_gain': {
                                method: float(np.mean(values)) for method, values in gains.items()}})
    return {'dataset': dataset, 'session': spec['session_id'], 'sample_rate_hz': spec['sample_rate_hz'],
            'inputs_sha256': hashes, 'baseline_sha256': spec['baseline_sha256'], 'windows': windows,
            'frames': sum(window['frames'] for window in windows), 'checks': checks, 'max_error': max_error}


def main():
    require_run_context()
    ledger = ARCHIVE / 'evidence-sha256.json'
    sources = read(ledger)
    rows = [audit_dataset(dataset, sources) for dataset in ('DS7', 'DS8', 'DS9')]
    output = {'schema': 'starlink-leo-pilot-archive/v1', 'source_ledger_sha256': sha256(ledger),
              'tolerance': TOLERANCE, 'datasets': rows}
    destination = experiment_output('E099')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'leo_pilot_archive.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    for row in rows:
        print(f'{row["dataset"]}: {row["frames"]} frames, {row["checks"]} checks, max error {row["max_error"]:.3g}')


if __name__ == '__main__':
    main()
