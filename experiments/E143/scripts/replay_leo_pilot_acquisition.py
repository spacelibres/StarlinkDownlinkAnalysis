"""Replay the reference project's public known-pilot detector from saved IQ."""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(os.environ.get('STARLINK_EXTERNAL_LEO_ROOT', '../leo-tracker-reduxredux-main')).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / 'src'))
    from leo.analysis.starlink.acquisition import ReceiverFrequencyCalibration, SymbolwiseAcquisitionConfig
    from leo.analysis.starlink.pilot_methods import PilotMethod, detect_pilot_methods
    from leo.analysis.starlink.templates import StarlinkEdge

    panel = Path('reports/2026_10_18_dsx01_pilot_clock')
    groups = ('DSX01-S05', 'paired_RX')
    inputs = {path.relative_to(root).as_posix(): digest(path) for path in
              sorted((root / 'src/leo/analysis/starlink').glob('*.py'))}
    inputs['src/leo/contracts/states.py'] = digest(root / 'src/leo/contracts/states.py')
    inputs['corpus/manifest.json'] = digest(root / 'corpus/manifest.json')
    calibration_label = b'E143 zero correction, no per-window receiver calibration'
    calibration = ReceiverFrequencyCalibration('uncalibrated', 0, hashlib.sha256(calibration_label).hexdigest())
    settings = {
        'default': SymbolwiseAcquisitionConfig(),
        'wide': SymbolwiseAcquisitionConfig(residual_cfo_min_hz=-600_000, residual_cfo_max_hz=600_000),
    }
    records = {}
    for group in groups:
        directory = panel / group
        selection = directory / 'selection.json'
        iq = directory / 'selected_iq.npz'
        for source in (selection, iq):
            inputs[source.as_posix()] = digest(root / source)
        metadata = json.loads((root / selection).read_text(encoding='utf-8'))
        rate = metadata['sample_rate_hz']
        with np.load(root / iq, allow_pickle=False) as samples:
            for row in metadata['windows']:
                index = row['number']
                values = samples[f'iq_{index}']
                sample_hash = hashlib.sha256(values.astype('<c8', copy=False).tobytes()).hexdigest()
                edge = StarlinkEdge(row['probe']['edge'])
                reference = row['winner']
                comparison = {
                    'group': group, 'window': index, 'receiver': row['receiver'],
                    'time_s': row['context_time_s'], 'historical_epoch': reference['integer_epoch_sample'],
                    'historical_tracking_cfo_hz': reference['fractional_tracking_cfo_hz'],
                }
                if sample_hash in records:
                    record = records[sample_hash]
                    if record['edge'] != edge or record['rate_hz'] != rate:
                        raise ValueError('Duplicate IQ has inconsistent acquisition metadata')
                    record['references'].append(comparison)
                    continue
                record = {'iq_sha256': sample_hash, 'edge': edge, 'rate_hz': rate,
                          'sample_count': len(values), 'references': [comparison], 'replays': {}}
                records[sample_hash] = record
                for name, config in settings.items():
                    detection = detect_pilot_methods(values, rate, sample_start=0, calibration=calibration,
                                                     acquisition_config=config, edge=edge)
                    score = next((score for score in detection.scores if score.method == PilotMethod.GLRT64), None)
                    record['replays'][name] = {
                        'status': detection.status, 'local_epoch': detection.local_epoch_sample,
                        'acquired_cfo_hz': detection.acquired_cfo_hz,
                        'tracking_cfo_hz': score.tracking_cfo_hz if score else None,
                        'glrt64_score': score.exact_score if score else None,
                        'rolled_control_score': score.control_score if score else None,
                        'margin': score.margin if score else None,
                    }
                print(group, index, record['replays'], flush=True)
    output = experiment_output('E143')
    output.mkdir(parents=True, exist_ok=True)
    result = {
        'schema': 'starlink-leo-pilot-acquisition-replay/v1', 'external_inputs_sha256': inputs,
        'receiver_correction_hz': 0, 'search_range_hz': {name: [config.residual_cfo_min_hz,
                                                            config.residual_cfo_max_hz]
                                                        for name, config in settings.items()},
        'numpy_version': np.__version__, 'unique_windows': list(records.values()),
    }
    (output / 'leo_pilot_acquisition.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
