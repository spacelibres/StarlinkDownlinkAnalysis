"""Contrast nine-tone signals with OFDM-rate recurrence in measured narrowband IQ."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import RAW_WAV, experiment_output, require_run_context
from audit_cross_capture_decoding import cadence
from audit_h2_projection import TRACK, SELECTION_ROOT
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_transfer import raw_phase
from import_starloc_ranges import DIR


SPACINGS = (43_945.5, 41_500., 42_500., 45_000., 47_000., 48_000.)
SLOPES = np.arange(-4500, 4501, 750)
LINES = np.arange(-4, 5)
SEARCH = 350_000
POWER_CAP = 30.
MIN_RATIO = 6.
CONTROL_FACTOR = 1.5
OUTER = (320_250., 879_750.)
TONE_WIDTH = 750.
LOCAL_MEDIAN_BINS = 256


class CombSearch:
    def __init__(self, fs, count, reference_tones=()):
        self.fs = fs
        self.nfft = max(4096, fft.next_fast_len(count, real=True))
        self.nfft = 1 << (self.nfft-1).bit_length()
        self.resolution = fs/self.nfft
        self.axis = fft.fftshift(fft.fftfreq(self.nfft, 1/fs))
        at = np.arange(count)-(count-1)/2
        self.kernel = np.hanning(count)*np.exp(-1j*np.pi*SLOPES[:, None]*(at/fs)**2)
        self.centers = np.flatnonzero(abs(self.axis) <= SEARCH)
        self.grid = [self.centers[:, None]+np.rint(LINES*g/self.resolution).astype(int) for g in SPACINGS]
        self.index = np.arange(self.nfft)
        self.median_grid = np.arange(0, self.nfft, LOCAL_MEDIAN_BINS)+(LOCAL_MEDIAN_BINS-1)/2
        self.excluded = abs(self.axis) < 350
        for reference in reference_tones:
            self.excluded |= abs(self.axis-reference) < 500

    def spectrum(self, x, k):
        z = fft.fftshift(fft.fft(x*self.kernel[k], n=self.nfft))
        power = abs(z)**2
        base = np.median(power.reshape(-1, LOCAL_MEDIAN_BINS), axis=1)
        ratio = power/np.maximum(np.interp(self.index, self.median_grid, base), 1e-30)
        ratio[self.excluded] = 1.
        return ratio

    def fit(self, x):
        fits = [None]*len(SPACINGS)
        best = [(-np.inf, -np.inf)]*len(SPACINGS)
        for k, slope in enumerate(SLOPES):
            raw = np.log(np.maximum(self.spectrum(x, k), 1e-30))
            logged = np.minimum(maximum_filter1d(raw, size=5), np.log(POWER_CAP))
            for i, (g, spacing) in enumerate(zip(self.grid, SPACINGS, strict=True)):
                z = logged[g].mean(axis=1)
                peak = z.max()
                tied = np.flatnonzero(z == peak)
                sharp = raw[g[tied]].mean(axis=1)
                j = int(tied[np.argmax(sharp)])
                rank = (peak, sharp.max())
                if rank > best[i]:
                    best[i] = rank
                    fits[i] = {'spacing_hz': spacing, 'center_hz': float(self.axis[self.centers[j]]),
                               'slope_hz_per_s': int(slope), 'geometric_ratio': float(np.exp(peak))}
        return fits

    def score(self, x, fits, time_shift=0.):
        cache = {}
        rows = []
        for fit in fits:
            slope = fit['slope_hz_per_s']
            if slope not in cache:
                cache[slope] = maximum_filter1d(self.spectrum(x, int(np.searchsorted(SLOPES, slope))), size=5)
            center = fit['center_hz']+slope*time_shift
            tones = center+LINES*fit['spacing_hz']
            idx = np.rint(tones/self.resolution).astype(int)+self.nfft//2
            ratio = cache[slope][idx]
            rows.append({'geometric_ratio': float(np.exp(np.log(np.clip(ratio, 1e-30, POWER_CAP)).mean())),
                         'line_ratios': ratio.tolist()})
        return rows


def comb_pass(rows):
    signal = rows[0]['geometric_ratio']
    return signal >= MIN_RATIO and signal >= CONTROL_FACTOR*max(row['geometric_ratio'] for row in rows[1:])


def timing(x, fs, mask, band):
    cut = fft.ifft(fft.fft(x)*mask)
    result, _ = cadence(cut, fs, band)
    envelope = abs(cut)**2
    spectrum = abs(fft.rfft((envelope-envelope.mean())*np.hanning(len(cut))))**2
    freqs = fft.rfftfreq(len(cut), 1/fs)
    df = freqs[1]
    near = abs(freqs-750.) <= df
    control = (abs(freqs-750.) > 2*df) & (abs(freqs-750.) <= max(150., 4*df))
    power = float(spectrum[near].max()/max(spectrum[control].max(), 1e-30))
    return {'one_frame': result['1.0'], 'half_frame_control': result['largest_half_frame_control'],
            'coherent_exact_frame_pass': abs(result['1.0']['peak_offset_samples']) <= 2 and
            result['1.0']['max_score'] > result['largest_half_frame_control'],
            'intensity_750_hz_to_neighbor_max': power}


def timing_views(x, fs, nominal, full):
    freq = fft.fftfreq(len(x), 1/fs)
    tones = nominal['center_hz']+LINES*SPACINGS[0]
    narrow = (abs(freq[:, None]-tones) < TONE_WIDTH).any(axis=1)
    outer = (abs(freq) >= OUTER[0]) & (abs(freq) < OUTER[1]) & ~narrow
    views = {'outer_tone_excised': timing(x, fs, outer, OUTER)}
    if full:
        lohi = (350., .44*fs)
        total = (abs(freq) > lohi[0]) & (abs(freq) < lohi[1])
        views['whole'] = timing(x, fs, total, lohi)
        views['narrow_grid_only'] = timing(x, fs, narrow & total, lohi)
    return views


def starloc(inputs):
    meta, data = DIR/'selection.json', DIR/'selected_iq.bin'
    metadata = read(meta)
    for path in (meta, data):
        inputs[path] = sha256(path)
    if inputs[data] != metadata['snapshot_sha256']:
        raise ValueError('StarLoc bundle SHA-256 mismatch')
    source = np.memmap(data, mode='r', dtype='<c8')
    groups = defaultdict(list)
    for w in metadata['windows']:
        x = source[w['snapshot_offset']//8:][:w['sample_count']]
        if w['sha256'] != hashlib.sha256(x.tobytes()).hexdigest():
            raise ValueError('StarLoc window SHA-256 mismatch')
        groups[w['start_seconds']].append((w, x))
    fs = metadata['sample_rate_hz']
    search = CombSearch(fs, len(x), (-100_000., 100_000.))
    output = []
    for seconds, rows in groups.items():
        if [w['receiver'] for w, _ in rows] != [0, 1, 2]:
            raise ValueError('Expected three synchronous receivers')
        fits = search.fit(rows[0][1])
        held = []
        for w, y in rows[1:]:
            scores = search.score(y, fits)
            nominal = fits[0]
            comparison = search.score(y, [{**nominal, 'slope_hz_per_s': value}
                                          for value in (0, -nominal['slope_hz_per_s'])])
            held.append({'receiver': w['receiver'], 'grid_scores': scores, 'pass': comb_pass(scores),
                         'zero_and_reversed_drift': comparison})
        output.append({'start_seconds': seconds, 'group': rows[0][0]['group'], 'rx0_grid_fits': fits,
                       'held_receivers': held, 'both_held_pass': all(row['pass'] for row in held),
                       'timing': [timing_views(y, fs, fits[0], True) for _, y in rows]})
        print('StarLoc', seconds, 's:', output[-1]['both_held_pass'], flush=True)
    return output


def compare_narrow(inputs):
    src = read(checked_artifact('v038-wav-shifted-lines', 'E040', 'wav_shifted_lines.json', inputs))
    for path in (RAW_WAV, ROOT/TRACK):
        inputs[path] = sha256(path)
    fs, raw = wavfile.read(RAW_WAV, mmap=True)
    n = fs//10
    times = np.arange(n)/fs
    with np.load(ROOT/TRACK, allow_pickle=False) as record:
        polynomial = record['quadratic_coefficients']
    wav = []
    search = CombSearch(fs, n//2)
    for key, selected in (('calibration', src['calibration_blocks'][:8]), ('held', src['heldout_blocks'][:8]),
                          ('post', src['post_blocks'])):
        for block in selected:
            start = block*fs//5
            values = raw[start:start+n].astype(float)
            x = (values[:, 0]+1j*values[:, 1])*np.exp(-1j*raw_phase(polynomial, times+start/fs))
            res = split_views(x, fs, search)
            wav.append({'block': block, 'group': key, **res})
    npz = {}
    for dataset in ('DSX01-S05', 'paired_RX'):
        file = SELECTION_ROOT/dataset/'selected_iq.npz'
        sel = file.with_name('selection.json')
        for path in (file, sel):
            inputs[path] = sha256(path)
        fs = read(sel)['sample_rate_hz']
        with np.load(file, allow_pickle=False) as iq:
            for name in iq.files:
                x = iq[name]
                digest = hashlib.sha256(x.tobytes()).hexdigest()
                if digest not in npz:
                    estimate = CombSearch(fs, len(x)//2)
                    npz[digest] = {'labels': [], **split_views(x, fs, estimate)}
                npz[digest]['labels'].append(f'{dataset}/{name}')
    return wav, list(npz.values())


def split_views(x, fs, estimator):
    count = len(x)//2
    fits = estimator.fit(x[:count])
    check = estimator.score(x[count:], fits, time_shift=count/fs)
    result = {'calibration_grids': fits, 'held_grid_scores': check, 'held_comb_pass': comb_pass(check)}
    result.update(timing_views(x, fs, fits[0], False))
    return result


def main():
    require_run_context()
    inputs = {}
    reference = starloc(inputs)
    wav, npz = compare_narrow(inputs)
    result = {'schema': 'starlink-starloc-narrowband/v1', 'inputs_sha256': {p.relative_to(ROOT).as_posix(): h for p,h in inputs.items()},
              'candidate_spacings_hz': SPACINGS, 'slope_grid_hz_per_s': SLOPES.tolist(), 'center_search_halfwidth_hz': SEARCH,
              'power_clip': POWER_CAP, 'pass_ratio': MIN_RATIO, 'pass_ratio_to_each_wrong_grid': CONTROL_FACTOR,
              'excluded_starloc_instrument_frequencies_hz': [-100_000, 0, 100_000],
              'tone_exclusion_halfwidth_hz': TONE_WIDTH, 'outer_band_absolute_hz': OUTER,
              'starLoc': reference, 'wav': wav, 'npz': npz}
    path = experiment_output('E054')
    path.mkdir(parents=True, exist_ok=True)
    (path/'starloc_narrowband.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Compared', len(reference), 'three-receiver windows,', len(wav), 'WAV windows, and', len(npz), 'unique NPZ windows', flush=True)


if __name__ == '__main__':
    main()
