"""Decode the Code 5 time tag, test light time, and compare a historical catalog."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
from sgp4.api import Satrec

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, raw_record_set
from audit_code5_orbit_state import LEO_FRAMES, RATE, file_hash, unpack_state
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


GPS_UNIX_EPOCH = 315_964_800
GPS_UTC_OFFSET = 18
NS = 1_000_000_000
C = 299_792_458
SITE = (30.283620, -97.739560, 0.0)
CATALOG = PROJECT_ROOT / "references/qin/qin_2025_tle_controls.json"
TARGET = 59784


def integer_at(word, bit, width):
    return (word >> bit) & ((1 << width) - 1)


def split_timestamp(word):
    seconds = integer_at(word, 348, 32)
    nanos = integer_at(word, 380, 30)
    if nanos >= NS:
        raise ValueError("Non-canonical nanoseconds")
    return seconds, nanos


def utc_label(seconds, nanos):
    t = datetime.fromtimestamp(seconds + GPS_UNIX_EPOCH - GPS_UTC_OFFSET, timezone.utc)
    return f"{t:%Y-%m-%dT%H:%M:%S}.{nanos:09d}Z"


def rows(words, states):
    extracted = []
    for row in states:
        data, observed = words[row["frame"]]
        length = data[0]
        packed = int.from_bytes(data[length:], "little")
        sec, nano = split_timestamp(packed)
        position, velocity = unpack_state(data[length:])
        if (length != row["directory_length"] or integer_at(packed, 421, 32) != row["counter"]
                or not np.array_equal(position, row["position_m"]) or not np.array_equal(velocity, row["velocity_m_s"])):
            raise ValueError(f"The authenticated epoch input does not match E094 at frame {row['frame']}")
        absolute = sec * NS + nano
        elapsed = absolute - sec // 15 * 15 * NS
        extracted.append({"frame": row["frame"], "gps_seconds": sec, "fractional_ns": nano,
                          "utc": utc_label(sec, nano), "gps_ns": absolute, "counter": row["counter"],
                          "clock_correction_from_15s_grid_ns": round(elapsed - row["counter"] / RATE * NS, 3),
                          "anchor_40": integer_at(packed, 480, 40), "anchor_40_matches_15s":
                          integer_at(packed, 480, 40) == (sec // 15 * 15 * 64 + 32),
                          "direct_epoch_bits": int(observed[length * 8 + 348:length * 8 + 410].sum())})
    return extracted


def stats(values):
    return {"n": len(values), "rms": float(np.sqrt(np.mean(values**2))), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def clock_step(decoded, frame_axis):
    axis = np.asarray(frame_axis)
    order = np.argsort(axis, kind="stable")
    utc = np.array([r["gps_ns"] for r in decoded], np.int64)[order]
    frames = axis[order] - axis[order][0]
    fitted_rate = np.linalg.lstsq(np.c_[np.ones(len(frames)), frames], utc - utc[0], rcond=None)[0][1]
    return {"frames": len(utc), "first_to_last_span_ns": int(utc[-1] - utc[0]),
            "fitted_step_ns_per_frame": float(fitted_rate), "step_offset_ppm": float((fitted_rate / (NS / RATE) - 1)*1e6)}


def clock_and_doppler(decoded, states, toa, beta, fit, held, complete, frames):
    position = np.array([r["position_m"] for r in states])
    velocity = np.array([r["velocity_m_s"] for r in states])
    gps_ns = np.array([r["gps_ns"] for r in decoded], dtype=np.int64)
    ts = (gps_ns - gps_ns[0]) / NS
    site = frames.geodetic_to_ecef_km(*SITE) * 1000
    wrong_site = frames.geodetic_to_ecef_km(SITE[0], SITE[1] + 1, SITE[2]) * 1000
    tests = {}
    fitted_residual = None
    for name, receiver in (("ut_reference", site), ("one_degree_east", wrong_site), ("no_light_time", None)):
        if receiver is None:
            geometric_light_time = np.zeros(len(ts))
        else:
            delta = position - receiver
            rho = np.linalg.norm(delta, axis=1)
            geometric_light_time = rho / C
            range_rate = np.sum(delta * velocity, axis=1) / rho
        error = toa - ts - geometric_light_time
        offset = float(np.median(error[fit]))
        residue_ns = (error - offset) * NS
        test = {"fit_constant_s": offset, "held_timing_residual_ns": stats(residue_ns[held])}
        if receiver is not None:
            freq_error = beta * C - range_rate
            bias = float(np.median(freq_error[fit]))
            test.update(doppler_fit_constant_m_s=bias,
                        doppler_held_without_bias_m_s=stats(freq_error[held]),
                        doppler_held_bias_removed_m_s=stats((freq_error - bias)[held]),
                        held_range_m=stats(rho[held]))
        if name == "ut_reference":
            fitted_residual = residue_ns
        tests[name] = test
    side = []
    for i, row in enumerate(states):
        if complete[i] and (not row["on_waveform_grid"] or row["content"] != row["arrival"]):
            side.append({"frame": row["frame"], "content_minus_arrival_frames": row["content"] - row["arrival"],
                         "on_arrival_grid": row["on_waveform_grid"], "timing_residual_ns": float(fitted_residual[i])})
    return {"definition": "TOA minus decoded elapsed time minus geometric range divided by c",
            "calibration_frames": [states[i]["frame"] for i in np.flatnonzero(fit)],
            "held_frames": [states[i]["frame"] for i in np.flatnonzero(held)],
            "site_lat_lon_height": SITE, "models": tests, "off_grid_or_content_lag": side,
            "frame_956_timing_residual_ns": float(fitted_residual[956])}


def compare_catalog(decoded, states, held, frames):
    utc_ns = np.array([r["gps_ns"] for r in decoded], dtype=np.int64)[held] + (GPS_UNIX_EPOCH - GPS_UTC_OFFSET)*NS
    position = np.array([r["position_m"] for r in states])[held]
    velocity = np.array([r["velocity_m_s"] for r in states])[held]
    results = []
    for source in read(CATALOG)["sources"]:
        raw = source["response_utf8"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != source["response_sha256"]:
            raise ValueError("Historical catalog response digest changed")
        records = json.loads(raw)[0]["orbital_data"]
        record, = records
        sat = Satrec.twoline2rv(record["tle_line1"], record["tle_line2"])
        if sat.satnum != record["satellite_id"]:
            raise ValueError("Catalog identifier differs from the orbital element lines")
        controls = (0, -18, 18) if sat.satnum == TARGET else (0,)
        for shift in controls:
            jd, fraction = frames.julian_day_from_utc_ns(utc_ns + shift * NS)
            errors, r, v = sat.sgp4_array(jd, fraction)
            if np.any(errors):
                raise ValueError(f"SGP4 failed for {sat.satnum}")
            p, q = frames.teme_to_ecef(r, v, frames.greenwich_mean_sidereal_time_rad(jd, fraction))
            results.append({"catalog_id": sat.satnum, "name": record["satellite_name"], "tle_epoch": record["epoch"],
                            "utc_shift_s": shift, "position_gap_m": stats(np.linalg.norm(p * 1000 - position, axis=1)),
                            "velocity_gap_m_s": stats(np.linalg.norm(q * 1000 - velocity, axis=1))})
    return results


def main():
    require_run_context()
    inputs = {}

    def art(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    base = "v010-code5-parity-02"
    old = art(base, "E012", "code5_observations.npz")
    with np.load(old, allow_pickle=False) as data:
        generator = data["parity_generator"]
    model = read(art("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    first, _ = archive_first_and_prior(art("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
                                       art("v078-archive-code5-links", "E080", "recovered_code5_words.npz"), pn, generator)
    e094 = read(art("v092-code5-orbit-state", "E094", "code5_orbit_state.json"))
    e092 = read(art("v090-code5-trajectories", "E092", "code5_record_trajectories.json"))
    binding = read(art("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json"))["overlap"]["pairs"]
    states = e094["archive"]["state_rows"]
    archive = rows(first, states)
    with np.load(art("v093-qin-time-source", "E095", "qin_time_doppler.npz"), allow_pickle=False) as data:
        if not np.array_equal(data["frame"], [r["frame"] for r in states]):
            raise ValueError("Archive Doppler and decoded state index differ")
        toa, beta = data["toa_s"], data["beta"]
    held_ids = set(e092["held_frames"])
    fit_ids = set(e092["calibration_frames"])
    complete = np.array([t["direct_epoch_bits"] == 62 and not r["missing_direct_record_bits"]
                         for t, r in zip(archive, states, strict=True)])
    regular = complete & np.array([r["on_waveform_grid"] and r["content"] == r["arrival"] for r in states])
    fit = regular & np.array([r["frame"] in fit_ids for r in states])
    held = regular & np.array([r["frame"] in held_ids for r in states])
    source = (EXTERNAL_LEO_ROOT / LEO_FRAMES).read_bytes()
    frames_hash = hashlib.sha256(source).hexdigest()
    if frames_hash != e094["leo_frame_model_sha256"]:
        raise ValueError("The external coordinate transform differs from E094")
    frames = ModuleType("leo_frames_e095")
    exec(compile(source, str(LEO_FRAMES), "exec"), frames.__dict__)
    inputs[CATALOG] = file_hash(CATALOG)

    raw = {}
    for key, path, meta, field in (
        ("480MSps", old, art(base, "E012", "code5_parity.json"), "hard_bits"),
        ("250MSps", art("v022-ut-replay", "E014", "ut_observations.npz"),
         art("v022-ut-replay", "E014", "ut_transfer.json"), "hard_code5_words"),
    ):
        all_words = raw_record_set(path, read(meta)["words"], field, pn, generator)
        words = {frame: w[0] for frame, w in all_words.items() if 0 in w}
        raw_states = e094["raw"][key]["state_rows"]
        times = rows(words, raw_states)
        raw[key] = {"rows": times, "arrivals": clock_step(times, [r["arrival"] for r in raw_states])}
    qin_by_id = {r["frame"]: r for r in archive}
    ut_by_id = {r["frame"]: r for r in raw["250MSps"]["rows"]}
    matched = [{"archive": item["qin_frame"], "raw": item["ut_candidate"],
                "raw_minus_archive_time_ns": ut_by_id[item["ut_candidate"]]["gps_ns"] - qin_by_id[item["qin_frame"]]["gps_ns"]}
               for item in binding if item["ut_candidate"] in ut_by_id]
    timings = clock_and_doppler(archive, states, toa, beta, fit, held, complete, frames)
    comparison = compare_catalog(archive, states, held, frames)
    report = {"schema": "starlink-code5-gps-epoch/v1", "gps_epoch_unix_s": GPS_UNIX_EPOCH, "gps_utc_s": GPS_UTC_OFFSET,
              "inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              "leo_source_sha256": frames_hash, "archive": archive, "raw": raw, "capture_overlap": matched,
              "archive_clock": clock_step(archive, [r["content"] for r in states]),
              "ground_link": timings, "historical_same_launch_catalog": comparison}
    out = experiment_output("E095")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_gps_epoch.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Decoded UTC:", archive[0]["utc"], archive[-1]["utc"], raw["480MSps"]["rows"][0]["utc"])
    print("Ground models:", timings["models"])
    print("Historical catalog median positions (m):", [(r["catalog_id"], r["utc_shift_s"], r["position_gap_m"]["median"])
                                                       for r in comparison])


if __name__ == "__main__":
    main()
