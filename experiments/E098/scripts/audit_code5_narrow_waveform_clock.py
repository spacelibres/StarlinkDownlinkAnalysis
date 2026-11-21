"""Acquire, equalize, and read Code 5 time using software-limited 25 MS/s IQ."""

from pathlib import Path
import json
import sys

import numpy as np
from scipy import fft, signal
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_clock_correction import CLOCK_BITS, SCALE_DENOMINATOR, projected_clock, timestamp_numerator
from audit_code5_gps_epoch import NS, utc_label
from audit_code5_orbit_state import file_hash
from audit_leo_edge_passband import lowpass_gain
from audit_leo_recording_coverage import NATIVE_GLRT
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, sinc_sample
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from ut_receiver_core import pss_reference


WIDE_FS = 480_000_000
FS = 25_000_000
PERIOD = FS / 750
SUBCARRIER_HZ = 234_375
SLOT_WIDE = 2112
SLOT_NARROW = 110
PREFIX = 64
USEFUL = 2048
COARSE_TRIM = 8
LAYOUT_LENGTH = 32
WORD_START = 228
WORD_BITS = LAYOUT_LENGTH * 8 + np.array(CLOCK_BITS)
CARRIERS = np.arange(405, 512)
TARGET_CARRIERS = NONPILOT[WORD_START + WORD_BITS]
PILOTS_TRAIN = np.arange(488, 496, 2) - CARRIERS[0]
PILOTS_TEST = np.arange(489, 496, 2) - CARRIERS[0]
BAND_CONDITIONS = ((0, 0), (0, 250_000), (0, 1_000_000), (-100_000, 0), (100_000, 0),
                   (-250_000, 0), (250_000, 0))
COARSE_FREQUENCIES = np.arange(-200_000, 200_001, 25_000)
GROUP_SEED = 2026112198
DECODER_PATHS = ("decode", "erase_511_control", "wrong_polarity_control", "cadence_recovered")
IQ = "data/raw/legacy_wideband/Starlink_Downlink_IQ.npy"
SSS = "references/qin/sssVec.mat"


def narrow_waveform(values, tune, rolloff):
    # Fourier resampling is exact for the selected periodic block and produces real 25 MS/s complex IQ.
    size = len(values) // 96 * 96
    time = np.arange(size) / WIDE_FS
    mixed = values[:size] * np.exp(-2j * np.pi * tune * time)
    spectrum = fft.fft(mixed)
    spectrum *= lowpass_gain(fft.fftfreq(size, 1 / WIDE_FS), FS, rolloff)
    return signal.resample(spectrum, size * 5 // 96, domain="freq")


def sync_templates(sss_frequency, tune, rolloff):
    spectrum = np.zeros(USEFUL, dtype=np.complex128)
    spectrum[LOADED % USEFUL] = sss_frequency[LOADED % 1024]
    sss = fft.ifft(spectrum)
    zero = np.zeros(960)
    joined = np.concatenate((zero, pss_reference(), sss[-PREFIX:], sss, zero))
    reference = narrow_waveform(joined, tune, rolloff)
    begin = len(zero) * 5 // 96
    return tuple(reference[begin + slot*SLOT_NARROW + COARSE_TRIM:begin + (slot + 1)*SLOT_NARROW - COARSE_TRIM]
                 for slot in (0, 1))


def acquire(narrow, begin, end, pss, sss):
    samples = narrow[begin:end]
    ts = np.arange(len(samples)) / FS
    energy = np.sqrt(np.maximum(signal.convolve(abs(samples)**2, np.ones(len(pss)), mode="valid"), 1e-20))
    pss_norm, sss_norm = np.linalg.norm(pss), np.linalg.norm(sss)
    best = None
    for frequency in COARSE_FREQUENCIES:
        corrected = samples * np.exp(-2j * np.pi * frequency * ts)
        a = abs(signal.correlate(corrected, pss, mode="valid", method="fft")) / (energy * pss_norm)
        b = abs(signal.correlate(corrected, sss, mode="valid", method="fft")) / (energy * sss_norm)
        combined = a[:-SLOT_NARROW] + b[SLOT_NARROW:]
        i = int(np.argmax(combined))
        score = float(combined[i])
        if best is None or score > best["score"]:
            fraction = 0.0
            if 0 < i < len(combined) - 1:
                l, m, r = combined[i - 1:i + 2]**2
                if l + r < 2 * m:
                    fraction = float(np.clip((l - r) / (2 * (l + r - 2 * m)), -.5, .5))
            best = {"epoch_25m": begin + i - COARSE_TRIM + fraction, "score": score, "cfo_hz": int(frequency),
                    "pss_score": float(a[i]), "sss_score": float(b[i + SLOT_NARROW])}
    return best


def symbol_pair(narrow, epoch, frequency, tune, scale):
    start = max(0, int(epoch) // 5 * 5 - 1000)
    region = narrow[start:start + 2500]
    region = region[:len(region) // 5 * 5]
    # Reconstruct an evaluation grid from the narrow signal alone. No discarded IQ re-enters the receiver.
    interpolated = signal.resample(region, len(region) * 96 // 5)
    interpolated *= np.exp(2j * np.pi * (tune - frequency) * np.arange(len(interpolated)) / WIDE_FS)
    result = []
    for slot in (1, 2):
        points = (epoch - start) * 96 / 5 + (slot * SLOT_WIDE + PREFIX + np.arange(USEFUL)) * scale
        result.append(fft.fft(sinc_sample(interpolated, points))[CARRIERS])
    return result


def read_time(equalized, pn, erase_511=False, polarity=1, counter_lsb=None):
    selected = equalized[TARGET_CARRIERS - CARRIERS[0]]
    hard = (selected.real * polarity < 0).astype(np.uint8)
    bits = np.zeros(1024, dtype=np.uint8)
    bits[WORD_BITS] = hard ^ pn[WORD_BITS]
    if counter_lsb is not None:
        bits[WORD_BITS[0]] = counter_lsb
    observed = np.zeros(1024, dtype=bool)
    observed[WORD_BITS] = True
    if erase_511:
        observed[WORD_BITS[TARGET_CARRIERS == 511]] = False
    clock = projected_clock(np.packbits(bits, bitorder="little").tobytes(), observed, LAYOUT_LENGTH)
    return {"hard_72": hard.tolist(), "anchor_count_correction": clock,
            "timestamp_27ths_ns": None if clock is None else timestamp_numerator(*clock)}


def fit_counter_origin(counter_high, ordinals):
    candidates = {2 * counter_high[0] - ordinals[0], 2 * counter_high[0] + 1 - ordinals[0]}
    for high, i in zip(counter_high[1:], ordinals[1:], strict=True):
        candidates.intersection_update((2 * high - i, 2 * high + 1 - i))
    return int(next(iter(candidates))) if len(candidates) == 1 else None


def waveform_arm(narrow, sss, tune, taper, pn):
    templates = sync_templates(sss, tune, taper)
    first = acquire(narrow, 0, int(PERIOD * 1.53), *templates)
    phase = (first["epoch_25m"] + PERIOD / 2) % PERIOD - PERIOD / 2
    if phase < -COARSE_TRIM:
        phase += PERIOD
    nominal = np.arange(phase, len(narrow) - 3 * SLOT_NARROW, PERIOD)
    rows = [acquire(narrow, max(0, int(e) - 75), int(e) + 320, *templates) for e in nominal]
    slope, intercept = np.polyfit(np.arange(len(rows)), [r["epoch_25m"] for r in rows], 1)
    scale = slope / PERIOD
    reference = np.array([pn_reference(int(k), np.array([2]))[0] for k in CARRIERS])
    equalized = []
    for row in rows:
        a, b = symbol_pair(narrow, row["epoch_25m"], row["cfo_hz"], tune, scale)
        base = b * reference.conj() / (a / sss[CARRIERS])
        gain = np.mean(base[PILOTS_TRAIN])
        z = base / gain
        row["reserved_pilot_evm"] = float(np.sqrt(np.mean(abs(z[PILOTS_TEST] - 1)**2)))
        row["decode"] = read_time(z, pn)
        row["erase_511_control"] = read_time(z, pn, erase_511=True)
        row["wrong_polarity_control"] = read_time(z, pn, polarity=-1)
        equalized.append(z)
    epochs = np.array([r["epoch_25m"] for r in rows])
    ordinals = np.rint((epochs - epochs[0]) / PERIOD).astype(int)
    groups = np.arange(len(rows)) // 2
    shuffled = np.random.default_rng(GROUP_SEED).permutation(np.unique(groups))
    fit = np.flatnonzero(np.isin(groups, shuffled[:len(shuffled) // 2]))
    high_bits = np.array([r["decode"]["hard_72"][1:14] for r in rows], dtype=np.uint8) ^ pn[WORD_BITS[1:14]]
    high_values = high_bits @ (1 << np.arange(13))
    origin = fit_counter_origin(high_values[fit], ordinals[fit])
    wrong_origin = fit_counter_origin(high_values[fit], ordinals[fit][::-1])
    for i, (row, z) in enumerate(zip(rows, equalized, strict=True)):
        inferred = origin + int(ordinals[i]) if origin is not None else -1
        if 0 <= inferred < 11250 and inferred // 2 == high_values[i]:
            row["cadence_recovered"] = read_time(z, pn, erase_511=True, counter_lsb=inferred & 1)
        else:
            row["cadence_recovered"] = {"hard_72": row["decode"]["hard_72"], "anchor_count_correction": None,
                                        "timestamp_27ths_ns": None}
    return {"first_sync": first, "estimated_period_25m": float(slope), "estimated_epoch_25m": float(intercept),
            "estimated_clock_ppm": float((scale - 1) * 1e6), "frames": rows,
            "cadence_calibration_ordinals": ordinals[fit].tolist(), "counter_origin": origin,
            "counter_origin_reversed_order": wrong_origin}


def compare(arm, expected, words):
    if len(arm["frames"]) != len(expected):
        raise ValueError("Found a different number of frame opportunities, cannot compare by ordinal")
    for r, witness in zip(arm["frames"], expected, strict=True):
        frame = witness["frame"]
        r["frame_index_after_reception"] = frame
        target = witness["gps_ns"] * SCALE_DENOMINATOR
        for path in DECODER_PATHS:
            values = r[path]
            prediction = values["timestamp_27ths_ns"]
            errors = np.array(values["hard_72"]) != words[frame][WORD_BITS]
            values["wrong_carriers_before_any_erasure"] = TARGET_CARRIERS[errors].tolist()
            values["timestamp_error_27ths_ns"] = None if prediction is None else prediction - target
            if prediction is not None:
                gps_s, nanos = divmod(prediction // SCALE_DENOMINATOR, NS)
                values["decoded_utc"] = utc_label(gps_s, nanos)


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    raw = PROJECT_ROOT / IQ
    reference = PROJECT_ROOT / SSS
    inputs[raw], inputs[reference] = file_hash(raw), file_hash(reference)
    legacy = "v010-code5-parity-02"
    provenance = read(art(legacy, "E012", "code5_parity.json"))
    if inputs[raw] != provenance["input_sha256"] or inputs[reference] != provenance["sss_reference_sha256"]:
        raise ValueError("Raw waveform or SSS reference changed")
    model = read(art("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    leo = read(art("v069-leo-recording-coverage", "E071", "leo_recording_coverage.json"))
    leo_hash = file_hash(EXTERNAL_LEO_ROOT / NATIVE_GLRT)
    if leo_hash != leo["sources_sha256"][NATIVE_GLRT.as_posix()]:
        raise ValueError("LEO tuning reference changed")
    edge = leo["separate_native_glrt_capture"]
    tuner = 511 * SUBCARRIER_HZ - edge["target_baseband_hz"]
    iq = np.load(raw, mmap_mode="r", allow_pickle=False)
    sss = loadmat(reference)["sssVecFull"].ravel()
    arms = {}
    for offset, taper in BAND_CONDITIONS:
        tune = tuner - offset
        values = narrow_waveform(iq, tune, taper)
        arm = waveform_arm(values, sss, tune, taper, pn)
        arms[f"{offset}/{taper}"] = {"center_hz": tune, "sample_count": len(values), **arm}
        print("Acquired", offset, taper, len(arm["frames"]), "frames", flush=True)
    # The truth timestamps and waveform decisions enter only after all receiver arms finish.
    expected = read(art("v093-code5-gps-epoch", "E095", "code5_gps_epoch.json"))["raw"]["480MSps"]["rows"]
    with np.load(art(legacy, "E012", "code5_observations.npz"), allow_pickle=False) as z:
        words = {w["frame"]: b for w, b in zip(provenance["words"], z["hard_bits"], strict=True)
                 if w["field"] == "first" and w["word"] == 0}
    for key, arm in arms.items():
        compare(arm, expected, words)
        passed = [sum(r[p]["timestamp_error_27ths_ns"] is not None and abs(r[p]["timestamp_error_27ths_ns"]) <= 27
                      for r in arm["frames"]) for p in DECODER_PATHS]
        print(key, "time labels within 1 ns:", passed, flush=True)
    output = {"schema": "starlink-code5-narrow-waveform-clock/v1", "samples_per_second": FS,
              "inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              "leo_native_source_sha256": leo_hash, "nominal_center_relative_to_input_hz": tuner,
              "cadence_group_seed": GROUP_SEED,
              "assumed_layout": {"length": LAYOUT_LENGTH, "first_codeword_start": WORD_START},
              "target_carriers": TARGET_CARRIERS.tolist(), "decoder_arms": arms}
    out = experiment_output("E098")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_narrow_waveform_clock.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
