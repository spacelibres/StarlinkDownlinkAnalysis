"""Check the external LEO Tracker lineage and re-evaluate stored dual-RX fits."""

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import shift
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, LEO_SNAPSHOT, experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import read


REPORT = Path('reports/2026_10_18_dsx01_pilot_clock')
BLOCKS = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12, 13))
BODY_SAMPLES = 3296  # Trim inside the 3300-sample nominal body at 2.5 MS/s.
SLOT_SECONDS = 4.4e-6
BIAS_LIMIT = 150.


def frame_product(a, b, sample_rate, epoch, delay, reference):
    shifted = shift(b, -delay, order=1, prefilter=False)
    x = a*shifted.conj()
    starts = np.ceil(epoch+(np.arange(14)/750+2*SLOT_SECONDS)*sample_rate).astype(int) + 2
    idx = starts[:, None]+np.arange(BODY_SAMPLES)
    if idx.min() < 0 or idx.max() >= len(x):
        raise ValueError('Invalid complete-frame coordinates')
    time = (idx-epoch)/sample_rate-7/750
    values = x[idx]*np.exp(-2j*np.pi*reference*time)
    return time, values


def transfer_grid(time, values, reference_slope):
    energy = (abs(values)**2).sum(axis=1)
    rows = []
    for left in combinations(range(len(BLOCKS)), 2):
        train = np.array([i for g in left for i in BLOCKS[g]])
        held = np.setdiff1d(np.arange(len(values)), train)
        models = {}
        for key, slope in (('archived', reference_slope), ('zero', 0.), ('reverse', -reference_slope)):
            deramped = values*np.exp(-1j*np.pi*slope*time**2)

            def energy_of_cfo(offset, idx):
                correction = np.exp(-2j*np.pi*offset*time[idx])
                return float((abs((correction*deramped[idx]).sum(axis=1))**2).sum()/values.shape[1])

            fit = minimize_scalar(lambda f: -energy_of_cfo(f, train), bounds=(-BIAS_LIMIT, BIAS_LIMIT), method='bounded')
            models[key] = {'trained_cfo_residual_hz': float(fit.x), 'slope_hz_per_s': slope,
                           'trained_explained_fraction': float(-fit.fun/energy[train].sum()),
                           'held_explained_fraction': float(energy_of_cfo(fit.x, held)/energy[held].sum()),
                           'near_bound': bool(abs(fit.x) > BIAS_LIMIT-.05)}
        rows.append({'trained_blocks': list(left), 'held_frames': held.tolist(), 'models': models})
    return rows


def main():
    require_run_context()
    inputs = {}
    captures = EXTERNAL_LEO_ROOT/REPORT
    for folder in ('DSX01-S05', 'paired_RX'):
        for name in ('selection.json', 'selected_iq.npz'):
            rel = REPORT/folder/name
            source = EXTERNAL_LEO_ROOT/rel
            snapshot = LEO_SNAPSHOT/rel
            inputs[rel.as_posix()] = sha256(source)
            if inputs[rel.as_posix()] != sha256(snapshot):
                raise ValueError('The LEO source changed relative to the local snapshot')
    for rel in (REPORT/'paired_RX/differential_chirp.json', Path('tools/probe_dsx01_pilot_clock.py'),
                Path('tools/dsx01_pilot_clock.py'), Path('corpus/manifest.json')):
        inputs[rel.as_posix()] = sha256(EXTERNAL_LEO_ROOT/rel)
    left = read(captures/'paired_RX/selection.json')
    archived = read(captures/'paired_RX/differential_chirp.json')
    rate = left['sample_rate_hz']
    rows = []
    with np.load(captures/'paired_RX/selected_iq.npz', allow_pickle=False) as data:
        for idx, obs in enumerate(archived):
            a, b = left['windows'][2*idx:2*idx+2]
            fields = ('actual_rf_hz', 'visit_index', 'valid_start_counter')
            if any(a['probe'][key] != b['probe'][key] for key in fields):
                raise ValueError('The captures do not share the same recorded support')
            epoch = a['winner']['integer_epoch_sample']+a['winner']['fractional_epoch_offset_samples']
            anchor = obs['template_free_cross_IQ']
            slope = anchor['difference_rate_hz_s']
            args = rate, epoch, obs['delay_samples'], anchor['frequency_difference_hz']
            measured = transfer_grid(*frame_product(data[f'iq_{2*idx}'], data[f'iq_{2*idx+1}'], *args), slope)
            wrong = transfer_grid(*frame_product(data[f'iq_{2*idx}'], data[f'iq_{3-2*idx}'], *args), slope)
            rows.append({'pair': idx, 'time_s': obs['time_s'], 'published_fit': anchor,
                         'delay_samples': obs['delay_samples'], 'actual_rf_hz': a['probe']['actual_rf_hz'],
                         'common_visit': a['probe']['visit_index'], 'measured': measured, 'different_time': wrong})
    out = experiment_output('E055')
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'starlink-leo-pair-transfer/v1', 'external_root': str(EXTERNAL_LEO_ROOT),
              'external_inputs_sha256': inputs, 'sample_rate_hz': rate, 'blocks': BLOCKS,
              'body_samples': BODY_SAMPLES, 'fitted_cfo_range_hz': [-BIAS_LIMIT, BIAS_LIMIT], 'pairs': rows}
    (out/'leo_pair_transfer.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Read', len(rows), 'dual-RX captures from external LEO Tracker; all source snapshots match', flush=True)


if __name__ == '__main__':
    main()
