"""Recheck common timing and per-layer observability across all local RF sources."""

import hashlib
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np
from scipy import fft, signal
from scipy.io import wavfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import RAW_WAV, experiment_output, require_run_context
from audit_h2_projection import (TRACK, WAV_MODEL, SELECTION_ROOT, affine_projection, band_masks,
                                 capture_records, header_bits, visible_positions)
from audit_ut_binary_tail import SOURCE, sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_cycle_transfer import raw_phase
from layout_core import parse_descriptors
from sparse_code5_core import binary_rank
from ut_receiver_core import read_ci16


HALF_WINDOW = 64
PERIODS = (1., 2., 3., 4., .5, 1.5, 2.5, 3.5)
FRAME_RATE = 750
LEGACY_RAW = "data/raw/legacy_wideband/Starlink_Downlink_IQ.npy"
CENTRAL = (320_250., 1_049_250.)
EDGE = (100_000., 1_150_000.)


def bandpass(samples, fs, band):
    frequencies = abs(fft.fftfreq(len(samples), 1/fs))
    lo, hi = band
    return fft.ifft(fft.fft(samples)*((frequencies > lo) & (frequencies < hi)))


def cadence(samples, fs, band):
    """Use only the cut IQ samples, without frame samples/timing/letter labels."""
    values = bandpass(samples, fs, band)
    correlation = signal.correlate(values, values, method="fft")
    energy = np.r_[0., np.cumsum(abs(values)**2, dtype=float)]
    summary, profiles = {}, {}
    for order in PERIODS:
        center = int(round(order*fs/FRAME_RATE))
        lags = center+np.arange(-HALF_WINDOW, HALF_WINDOW+1)
        norm = np.sqrt((energy[-1]-energy[lags])*energy[len(values)-lags])
        scores = abs(correlation[len(values)-1+lags])/np.maximum(norm, 1e-30)
        peak = int(np.argmax(scores))
        summary[str(order)] = {"nearest_lag_samples": center, "peak_offset_samples": int(lags[peak]-center),
                               "at_nearest": float(scores[HALF_WINDOW]), "max_score": float(scores[peak]),
                               "window_median": float(np.median(scores))}
        profiles[str(order)] = scores.tolist()
    summary["largest_half_frame_control"] = max(summary[str(x)]["max_score"] for x in PERIODS[4:])
    return summary, profiles


def survey_wav(groups, inputs):
    paths = (RAW_WAV, ROOT/TRACK)
    for p in paths:
        inputs[p] = sha256(p)
    rate, raw = wavfile.read(RAW_WAV, mmap=True)
    with np.load(paths[1], allow_pickle=False) as track:
        polynomial = track["quadratic_coefficients"]
    width = rate//5
    rows = []
    times = np.arange(width)/rate
    for name, block_ids in groups.items():
        for block in block_ids:
            first = block*width
            chunk = raw[first:first+width].astype(float)
            z = (chunk[:, 0]+1j*chunk[:, 1])*np.exp(-1j*raw_phase(polynomial, times+first/rate))
            stats, profiles = cadence(z, rate, CENTRAL)
            rows.append({"group": name, "block": block, "first_sample": first, "cadence": stats, "profiles": profiles})
    return {"sample_rate": rate, "band_absolute_hz": CENTRAL, "rows": rows}


def survey_npz(inputs):
    unique = {}
    for dataset in ("DSX01-S05", "paired_RX"):
        iq_file = SELECTION_ROOT/dataset/"selected_iq.npz"
        meta = iq_file.with_name("selection.json")
        inputs[iq_file] = sha256(iq_file)
        inputs[meta] = sha256(meta)
        rate = read(meta)["sample_rate_hz"]
        with np.load(iq_file, allow_pickle=False) as archive:
            for name in archive.files:
                y = archive[name]
                key = hashlib.sha256(y.tobytes()).hexdigest()
                label = f"{dataset}/{name}"
                if key in unique:
                    unique[key]["labels"].append(label)
                    continue
                stats, profiles = cadence(y, rate, EDGE)
                unique[key] = {"labels": [label], "iq_sha256": key, "sample_rate": rate, "cadence": stats, "profiles": profiles}
    return {"band_absolute_hz": EDGE, "rows": list(unique.values())}


def survey_wide_projection(inputs):
    rows = []
    for name, path, fs, cfo in (("ut250", ROOT/SOURCE, 250_000_000, -171_800.),
                                ("legacy480", ROOT/LEGACY_RAW, 480_000_000, 48_258.)):
        inputs[path] = sha256(path)
        y = read_ci16(path) if name == "ut250" else np.load(path, mmap_mode="r", allow_pickle=False)
        # Nonoverlapping complete 10 ms records. A 4T lag remains visible.
        length = fs//100
        for index in range(len(y)//length):
            n = np.arange(length)
            cut = np.asarray(y[index*length:(index+1)*length])
            for view, center, rate, band in (("central", cfo-123_416., 2_400_000, CENTRAL),
                                               ("lower_edge", cfo-492.5*234_375, 2_500_000, EDGE)):
                divisor = gcd(fs, rate)
                shifted = cut*np.exp(-2j*np.pi*center*n/fs)
                projected = signal.resample_poly(shifted, rate//divisor, fs//divisor)
                stats, profiles = cadence(projected, rate, band)
                rows.append({"source": name, "window": index, "source_first_sample": index*length,
                             "view": view, "output_center_relative_hz": center, "output_rate": rate,
                             "band_absolute_hz": band, "cadence": stats, "profiles": profiles})
    return rows


def code_ranks(reports, generator, inputs):
    # These are conditional ideal carrier masks used by E024; simultaneous
    # views of the same header are not assumed.
    for p in (ROOT/WAV_MODEL, ROOT/TRACK, *(SELECTION_ROOT/name/"selection.json" for name in ("DSX01-S05", "paired_RX"))):
        inputs[p] = sha256(p)
    masks, meta = band_masks()
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    outgoing, cache = [], {}
    for capture, report in reports.items():
        for index, h1, h2 in capture_records(report, capture):
            first_bits = header_bits(h1["words_hex"])
            second_bits = header_bits(h2["words_hex"])
            start = h2["start_resource"]
            h1_length = 114*(len(first_bits)//32)
            h2_length = 114*(len(second_bits)//32)
            first_entries = parse_descriptors(first_bits, 20)["entries"]
            if not first_entries or first_entries[0]["mcs_candidate"] != 5 or h1_length+2048 > start:
                raise ValueError("The first Code 5 region is not available")
            h1_adjust = tuple(range(16, 16+20*len(first_entries)))
            h2_adjust = tuple(range(16, 16+20*len(parse_descriptors(second_bits,20)["entries"])))
            rows = {}
            for label, carriers in masks.items():
                if label in ("npz_union", "optimistic_wav_npz_union"):
                    continue
                p1 = visible_positions(0, h1_length, carriers)
                p2 = visible_positions(start, h2_length, carriers)
                p5 = visible_positions(h1_length, 2048, carriers)
                if p5 not in cache:
                    cache[p5] = binary_rank(code[list(p5)])
                rows[label] = {"H1": affine_projection(first_bits, h1_adjust, p1),
                               "H2": affine_projection(second_bits, h2_adjust, p2),
                               "Code5": {"observed_coordinates": len(p5), "observable_rank": cache[p5],
                                         "free_dimensions": 1024-cache[p5]}}
            outgoing.append({"capture": capture, "frame_index": index, "h2_start": start, "views": rows})
    return {"mask_metadata": meta, "wideband_h1_h2_crc_frames": len(outgoing), "projected_frames": outgoing}


def main():
    require_run_context()
    inputs = {}
    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    block_source = read(artifact("v038-wav-shifted-lines", "E040", "wav_shifted_lines.json"))
    groups = {"calibration": block_source["calibration_blocks"], "held": block_source["heldout_blocks"],
              "post": block_source["post_blocks"]}
    reports = {"legacy480": read(artifact("v008-complete-header-layout-02", "E010", "full_header_layout.json")),
               "ut250": read(artifact("v022-ut-replay", "E014", "ut_transfer.json"))}
    code = artifact("v010-code5-parity-02", "E012", "code5_observations.npz")
    with np.load(code, allow_pickle=False) as data:
        ranks = code_ranks(reports, data["parity_generator"], inputs)
    report = {"schema": "starlink-cross-capture-decoding-audit/v1", "nominal_rate_hz": FRAME_RATE,
              "wav": survey_wav(groups, inputs), "npz": survey_npz(inputs), "wide_projections": survey_wide_projection(inputs),
              "information_rank": ranks, "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
              "scope": "H1/H2 and Code5 ranks use ideal noiseless carrier masks. Derived views are not extra RF captures."}
    output = experiment_output("E047")
    output.mkdir(parents=True, exist_ok=True)
    (output/"cross_capture_decoding.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Compared WAV, NPZ and both wideband projections with", ranks["wideband_h1_h2_crc_frames"], "header witnesses.")


if __name__ == "__main__":
    main()
