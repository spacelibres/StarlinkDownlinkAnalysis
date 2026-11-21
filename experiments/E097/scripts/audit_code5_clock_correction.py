"""Verify the Code 5 redundant frame clock and narrowband time reconstruction."""

from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, raw_record_set
from audit_code5_gps_epoch import NS, integer_at, split_timestamp
from audit_code5_orbit_state import file_hash
from audit_code5_record_trajectories import crc_rows, field_information, interval_bits
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


COUNTER = (421, 14)
CORRECTION = (453, 23)
ANCHOR = (485, 35)
TARGETS = {"counter_low": COUNTER, "correction": CORRECTION, "interval_anchor": ANCHOR}
FRAME_TIME_BITS = tuple(range(COUNTER[0], sum(COUNTER))) + tuple(range(CORRECTION[0], sum(CORRECTION)))
CLOCK_BITS = FRAME_TIME_BITS + tuple(range(ANCHOR[0], sum(ANCHOR)))
DIRECT_TIME = (348, 62)
FRAME_RATE_HZ = 750
INTERVAL_S = 15
CLOCK_TICKS_PER_S = 540_000_000
CONSTANT_TICKS = 4_719_600
SCALE_DENOMINATOR = 27
LEO_SOURCE = Path("docs/research/starlink-literature/identity-timing-orbit.md")


def fields(packed):
    anchor = integer_at(packed, *ANCHOR)
    base, half = divmod(anchor, 2)
    counter = integer_at(packed, *COUNTER)
    if half != 1 or base % INTERVAL_S or counter >= INTERVAL_S * FRAME_RATE_HZ:
        return None
    return base, counter, integer_at(packed, *CORRECTION)


def projected_clock(data, observed, length):
    first = 8 * length
    if not observed[[first + bit for bit in FRAME_TIME_BITS]].all():
        return None
    missing = [bit for bit in range(ANCHOR[0], sum(ANCHOR)) if not observed[first + bit]]
    if len(missing) > 1:
        return None
    packed = int.from_bytes(data[length:], "little")
    if not missing:
        return fields(packed)
    bit = 1 << missing[0]
    candidates = [result for x in (packed & ~bit, packed | bit) if (result := fields(x)) is not None]
    return candidates[0] if len(candidates) == 1 else None


def timestamp_numerator(base, counter, correction):
    # Convert the 540 MHz ticks to nanoseconds with 50/27 and retain the fraction.
    ticks = counter * (CLOCK_TICKS_PER_S // FRAME_RATE_HZ) + CONSTANT_TICKS - correction
    return base * NS * SCALE_DENOMINATOR + ticks * 50


def decode_clock_rows(words, times):
    if set(words) != {r["frame"] for r in times}:
        raise ValueError("Clock and codeword membership differ")
    rows = []
    for time in times:
        frame = time["frame"]
        data, observed = words[frame]
        length = data[0]
        packed = int.from_bytes(data[length:], "little")
        values = fields(packed)
        if values is None:
            raise ValueError(f"Invalid frame counter or 15-second anchor at frame {frame}")
        base, counter, adjust = values
        s, ns = split_timestamp(packed)
        if (s, ns) != (time["gps_seconds"], time["fractional_ns"]) or integer_at(packed, 421, 32) != time["counter"]:
            raise ValueError(f"Clock fields differ from verified E095 input at frame {frame}")
        actual = s * NS + ns
        numerator = timestamp_numerator(base, counter, adjust)
        rows.append({"frame": frame, "length": length, "base_gps_s": base, "counter": counter,
                     "correction": adjust, "timing_selector": integer_at(packed, 417, 4),
                     "before_selector": integer_at(packed, 410, 7),
                     "bits_476_to_484": integer_at(packed, 476, 9),
                     "direct_clock_bits": int(observed[[length * 8 + bit for bit in CLOCK_BITS]].sum()),
                     "direct_timestamp_bits": int(observed[length * 8 + 348:length * 8 + 410].sum()),
                     "observed_phase_thirds_ns": 3 * (actual - base * NS) - 4_000_000 * counter,
                     "timestamp_residual_27ths_ns": actual * SCALE_DENOMINATOR - numerator,
                     "floor_residual_ns": actual - numerator // SCALE_DENOMINATOR})
    return rows


def describe(arr):
    return {"min": float(np.min(arr)), "max": float(np.max(arr)), "rms": float(np.sqrt(np.mean(np.square(arr))))}


def controls(rows, calibrate, held):
    q = np.array([r["correction"] for r in rows])
    # Time perturbation computed in exact relative units first.
    phase = np.array([r["observed_phase_thirds_ns"] for r in rows], dtype=np.float64) / 3
    coeff = np.linalg.lstsq(np.c_[phase[calibrate], np.ones(calibrate.sum())], q[calibrate], rcond=None)[0]
    rate_models = {}
    for rate in (250_000_000, 480_000_000, 539_000_000, CLOCK_TICKS_PER_S, 541_000_000):
        scale = rate / NS
        offset = np.median(q[calibrate] + scale * phase[calibrate])
        predictions = (offset - q[held]) / scale
        rate_models[str(rate)] = {"held_residual_ns": describe(phase[held] - predictions),
                                  "intercept_fitted_in_ticks": float(offset)}
    return {"fitted_ticks_per_second": float(-coeff[0] * NS), "fitted_intercept_ticks": float(coeff[1]),
            "rate_controls": rate_models}


def observability(windows, lengths, generator, polynomial):
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    unknown = interval_bits(8, 1024)
    masks = [int.from_bytes(x.tobytes(), "little") & unknown for x in np.packbits(code, axis=1, bitorder="little")]
    constraints = {length: [r & unknown for r in crc_rows(length, polynomial)] for length in lengths}
    out = []
    for window in windows:
        observed = [masks[i] for i in window["codeword_coordinates"]]
        results = []
        for length in lengths:
            first = 8 * length
            fields = {k: interval_bits(first + s, first + s + width) for k, (s, width) in TARGETS.items()}
            fields["absolute_gps_time"] = interval_bits(first + DIRECT_TIME[0], first + sum(DIRECT_TIME))
            fields["redundant_clock"] = sum(fields[k] for k in TARGETS)
            results.append({"length": length, "targets": {k: {"rank": field_information(observed, [], mask),
                            "rank_with_crc": field_information(observed, constraints[length], mask)} for k, mask in fields.items()}})
        out.append({"kind": window["kind"], "labels": window["labels"], "layouts": results})
    return out


def replay_at_25msps(words, labels, positions, target_length, absolute):
    visible = np.zeros(1024, bool)
    visible[[i for i in positions if i < 1024]] = True
    results = []
    for row in labels:
        data, mask = words[row["frame"]]
        length = data[0]
        if length != target_length:
            continue
        # Eliminate every unavailable systematic bit, including the native time tag.
        actual_observation = mask & visible
        start = length * 8 + DIRECT_TIME[0]
        native_time = slice(start, start + DIRECT_TIME[1])
        excluded_timestamp_bits = int(actual_observation[native_time].sum())
        actual_observation[native_time] = False
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
        selected_data = np.packbits(bits & actual_observation, bitorder="little").tobytes()
        if absolute:
            decoded = projected_clock(selected_data, actual_observation, length)
        else:
            decoded = None
            if actual_observation[[8 * length + bit for bit in FRAME_TIME_BITS]].all():
                packed = int.from_bytes(selected_data[length:], "little")
                number = integer_at(packed, *COUNTER)
                if number < INTERVAL_S * FRAME_RATE_HZ:
                    decoded = (0, number, integer_at(packed, *CORRECTION))
        numerator = None if decoded is None else timestamp_numerator(*decoded)
        target = row["gps_ns"]
        if not absolute:
            target %= INTERVAL_S * NS
        results.append({"frame": row["frame"],
                        "native_timestamp_bits_visible_but_unused": excluded_timestamp_bits,
                        "missing_required_record_bits": [i for i in (CLOCK_BITS if absolute else FRAME_TIME_BITS)
                                                         if not actual_observation[8 * length + i]],
                       "reconstructed_floor_time_ns": None if numerator is None else numerator // SCALE_DENOMINATOR,
                       "rational_residual_27ths_ns": None if numerator is None else target * SCALE_DENOMINATOR - numerator})
    return results


def main():
    require_run_context()
    inputs = {}

    def art(run, e, name):
        return checked_artifact(run, e, name, inputs)

    legacy = "v010-code5-parity-02"
    legacy_iq = art(legacy, "E012", "code5_observations.npz")
    with np.load(legacy_iq, allow_pickle=False) as z:
        generator = z["parity_generator"]
    model = read(art("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    archive, _ = archive_first_and_prior(art("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
                                         art("v078-archive-code5-links", "E080", "recovered_code5_words.npz"), pn, generator)
    e095 = read(art("v093-code5-gps-epoch", "E095", "code5_gps_epoch.json"))
    e092 = read(art("v090-code5-trajectories", "E092", "code5_record_trajectories.json"))
    e094 = read(art("v092-code5-orbit-state", "E094", "code5_orbit_state.json"))
    checks = read(art("v089-code5-directory-final", "E091", "code5_directory_checks.json"))
    qin = decode_clock_rows(archive, e095["archive"])
    regular = np.array([r["direct_clock_bits"] == len(CLOCK_BITS) and r["direct_timestamp_bits"] == 62 for r in qin])
    ids = np.array([r["frame"] for r in qin])
    calibrate = np.isin(ids, e092["calibration_frames"]) & regular
    held = np.isin(ids, e095["ground_link"]["held_frames"]) & regular
    metrics = controls(qin, calibrate, held)
    metrics.update(calibration_frames=ids[calibrate].tolist(), held_frames=ids[held].tolist())
    wrong_arrival = []
    for q, state in zip(qin, e094["archive"]["state_rows"], strict=True):
        skip = state["content"] - state["arrival"]
        if skip:
            error = q["timestamp_residual_27ths_ns"] + skip * 36_000_000
            wrong_arrival.append({"frame": state["frame"], "count_delta": skip, "error_using_arrival_count_ns": error / 27})

    upper, = [w for w in e092["narrowband"] if w["kind"] == "catalogued_nominal" and len(w["labels"]) == 6]
    lower, = [w for w in e092["narrowband"] if w["kind"] == "catalogued_nominal" and len(w["labels"]) == 20]
    projection = {"qin_2025": replay_at_25msps(archive, e095["archive"], upper["codeword_coordinates"], 32, True)}
    partial = replay_at_25msps(archive, e095["archive"], lower["codeword_coordinates"], 48, False)
    raw = {}
    for key, path, meta, array in (
        ("480MSps", legacy_iq, art(legacy, "E012", "code5_parity.json"), "hard_bits"),
        ("250MSps", art("v022-ut-replay", "E014", "ut_observations.npz"),
         art("v022-ut-replay", "E014", "ut_transfer.json"), "hard_code5_words"),
    ):
        received = raw_record_set(path, read(meta)["words"], array, pn, generator)
        first = {i: w[0] for i, w in received.items() if 0 in w}
        raw[key] = decode_clock_rows(first, e095["raw"][key]["rows"])
        if key == "480MSps":
            projection[key] = replay_at_25msps(first, e095["raw"][key]["rows"], upper["codeword_coordinates"], 32, True)

    lengths = sorted({r["length"] for r in qin})
    projected_bands = observability(e092["narrowband"], lengths, generator, checks["crc_polynomial_including_top_bit"])
    report = {"schema": "starlink-code5-clock-correction/v1", "record_fields": TARGETS,
              "clock_ticks_per_second": CLOCK_TICKS_PER_S, "constant_ticks": CONSTANT_TICKS,
              "inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              "leo_literature_source_sha256": file_hash(EXTERNAL_LEO_ROOT / LEO_SOURCE),
              "archive": qin, "raw": raw, "clock_fit_and_controls": metrics,
              "arrival_vs_payload_clock": wrong_arrival, "ideal_band_coverage": projected_bands,
              "lower_25msps_relative_projection": {"labels": lower["labels"], "measured_codeword_projections": partial},
              "upper_25msps_projection": {"labels": upper["labels"], "measured_codeword_projections": projection}}
    output = experiment_output("E097")
    output.mkdir(parents=True, exist_ok=True)
    (output / "code5_clock_correction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, data in {"qin_2025": qin, **raw}.items():
        residual = [r["timestamp_residual_27ths_ns"] for r in data]
        print(key, len(data), min(residual), max(residual), "in 1/27 ns")
    for key, data in projection.items():
        print(key, len(data), sum(r["rational_residual_27ths_ns"] is not None for r in data), "complete projections")
    print("Lower-edge relative time:", len(partial), sum(r["rational_residual_27ths_ns"] is not None for r in partial))


if __name__ == "__main__":
    main()
