"""Read the packed Code 5 state vectors and test their orbital dynamics."""

from pathlib import Path
import json
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, crc_bytes, raw_record_set
from audit_code5_frame_count import RELATIVE_BIT
from audit_code5_record_trajectories import crc_rows, field_information, interval_bits
from audit_leo_copy_transfer import file_hash
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


STATE_START = 60
VELOCITY_START = 252
STATE_END = 348
RATE = 750
GM = 3.986004418e14
EARTH_A = 6_378_137.0
EARTH_J2 = 1.08262668e-3
EARTH_OMEGA = 7.292115e-5
OMEGA = np.array([0, 0, EARTH_OMEGA])
LEO_FRAMES = Path("src/leo/sky/frames.py")


def unpack_state(record, start=STATE_START, format_string="<3d3f"):
    width = STATE_END - STATE_START
    if len(record) * 8 < start + width:
        raise ValueError("Record does not cover the full state")
    value = (int.from_bytes(record, "little") >> start) & ((1 << width) - 1)
    values = np.array(struct.unpack(format_string, value.to_bytes(width // 8, "little")))
    return values[:3], values[3:]


def accelerations(position, velocity):
    radius = np.linalg.norm(position, axis=1)
    central = -GM * position / radius[:, None]**3
    z2 = (position[:, 2] / radius)**2
    j2_scale = 1.5 * EARTH_J2 * GM * EARTH_A**2 / radius**5
    j2 = j2_scale[:, None] * position * (5 * z2[:, None] - np.array([1, 1, 3]))
    coriolis = -2 * np.cross(OMEGA, velocity)
    centrifugal = -np.cross(OMEGA, np.cross(OMEGA, position))
    return {"central": central, "central_j2": central + j2,
            "earth_fixed": central + coriolis + centrifugal,
            "earth_fixed_j2": central + j2 + coriolis + centrifugal,
            "reverse_earth_rotation_j2": central + j2 - coriolis + centrifugal}


def norm_rmse(vectors):
    return float(np.sqrt(np.mean(np.sum(vectors**2, axis=1))))


def compare_pairs(state, left, right, clock="content", frame_rate=RATE):
    t = np.array([r[clock] for r in state], dtype=np.int64)
    position = np.array([r["position_m"] for r in state])
    velocity = np.array([r["velocity_m_s"] for r in state])
    dt = (t[right] - t[left]) / frame_rate
    if not len(dt) or np.any(dt <= 0):
        raise ValueError("Dynamics requires positive frame intervals")
    mid_position = (position[right] + position[left]) * .5
    mid_velocity = (velocity[right] + velocity[left]) * .5
    v_diff = (position[right] - position[left]) / dt[:, None]
    a_diff = (velocity[right] - velocity[left]) / dt[:, None]
    a_model = accelerations(mid_position, mid_velocity)
    return {"pairs": len(dt), "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
            "implied_state_frames_per_second": float(frame_rate * np.sum(v_diff * mid_velocity) / np.sum(v_diff**2)),
            "velocity_rmse_m_s": norm_rmse(v_diff - mid_velocity),
            "acceleration_rmse_m_s2": {key: norm_rmse(a_diff - a) for key, a in a_model.items()},
            "ecef_j2_beats_inertial_j2_pairs": int(np.sum(np.linalg.norm(a_diff - a_model["earth_fixed_j2"], axis=1)
                                                         < np.linalg.norm(a_diff - a_model["central_j2"], axis=1)))}


def adjacent(state, indices, clock):
    t = np.array([r[clock] for r in state])
    order = indices[np.argsort(t[indices], kind="stable")]
    valid = np.diff(t[order]) > 0
    return order[:-1][valid], order[1:][valid]


def group_endpoints(state, indices):
    groups = sorted({state[i]["group"] for i in indices})
    pairs = []
    for group in groups:
        members = sorted((int(i) for i in indices if state[i]["group"] == group),
                         key=lambda i: state[i]["content"])
        if state[members[0]]["content"] < state[members[-1]]["content"]:
            pairs.append((members[0], members[-1]))
    return np.array(pairs).T


def state_rows(words, crc_receipts, timestamps, directory):
    if set(words) != {r["frame"] for r in crc_receipts}:
        raise ValueError("The codeword and directory inventories differ")
    affine = directory["affine_convention"]
    rows = []
    format_controls = {key: 0 for key in ("little_endian", "big_endian", "one_bit_early", "one_bit_late")}
    for receipt in crc_receipts:
        frame = receipt["frame"]
        data, mask = words[frame]
        length = data[0]
        crc = crc_bytes(data[:length - 3], directory["crc_polynomial_including_top_bit"],
                        affine["seed"], affine["residue"])
        if (not receipt["crc_pass"] or length != receipt["directory_length"] or crc != data[length - 3:length]
                or crc.hex() != receipt["expected_crc"]):
            raise ValueError(f"Directory fails the recorded CRC in frame {frame}")
        pos, vel = unpack_state(data[length:])
        for label, shift, fmt in (("little_endian", 0, "<3d3f"), ("big_endian", 0, ">3d3f"),
                                  ("one_bit_early", -1, "<3d3f"), ("one_bit_late", 1, "<3d3f")):
            r, v = unpack_state(data[length:], STATE_START + shift, fmt)
            maxima = np.max(abs(r)), np.max(abs(v))
            if 3.4e6 <= maxima[0] <= 8e6 and 3.4e3 <= maxima[1] <= 9e3:
                format_controls[label] += int(6.4e6 <= np.linalg.norm(r) <= 8e6 and
                                              6.5e3 <= np.linalg.norm(v) <= 9e3)
        t = timestamps[frame]
        counter = (int.from_bytes(data[length:], "little") >> RELATIVE_BIT) & 0xFFFFFFFF
        if counter != t["counter"]:
            raise ValueError(f"State and verified frame count are inconsistent in frame {frame}")
        start = length * 8 + STATE_START
        missing = np.flatnonzero(~mask[start:length * 8 + STATE_END]) + STATE_START
        rows.append({"frame": frame, "directory_length": length, "position_m": pos.tolist(),
                     "velocity_m_s": vel.tolist(), "missing_direct_record_bits": missing.tolist(), **t})
    return rows, format_controls


def magnitude_ranges(state):
    pos = np.array([r["position_m"] for r in state])
    vel = np.array([r["velocity_m_s"] for r in state])
    inertial = vel + np.cross(OMEGA, pos)
    h = np.cross(pos, inertial)
    inc = np.rad2deg(np.arccos(h[:, 2] / np.linalg.norm(h, axis=1)))
    values = {"geocentric_radius_m": np.linalg.norm(pos, axis=1),
              "rotating_speed_m_s": np.linalg.norm(vel, axis=1), "inertial_speed_m_s": np.linalg.norm(inertial, axis=1),
              "inclination_deg": inc}
    return {key: [float(v.min()), float(v.max())] for key, v in values.items()}


def coverage(windows, lengths, generator, poly):
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    unknown = interval_bits(8, 1024)
    packed = [int.from_bytes(row.tobytes(), "little") & unknown
              for row in np.packbits(code, axis=1, bitorder="little")]
    constraints = {length: [r & unknown for r in crc_rows(length, poly)] for length in lengths}
    results = []
    for window in windows:
        coords = window["codeword_coordinates"]
        observations = [packed[i] for i in coords]
        bounds = []
        for length in lengths:
            base = 8 * length
            targets = {"position": (STATE_START, VELOCITY_START), "velocity": (VELOCITY_START, STATE_END),
                       "combined": (STATE_START, STATE_END)}
            summary = {}
            for key, (start, end) in targets.items():
                target = interval_bits(base + start, base + end)
                direct = [i - base for i in coords if base + start <= i < base + end]
                summary[key] = {"direct_record_bits": direct, "rank_without_crc": field_information(observations, [], target),
                                "rank_with_crc": field_information(observations, constraints[length], target)}
            bounds.append({"length": length, **summary})
        results.append({"kind": window["kind"], "labels": window["labels"], "bounds": bounds})
    return results


def main():
    require_run_context()
    inputs = {}

    def art(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    old = "v010-code5-parity-02"
    old_words = art(old, "E012", "code5_observations.npz")
    with np.load(old_words, allow_pickle=False) as data:
        generator = data["parity_generator"]
    model = read(art("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    archived = "v078-archive-code5-links"
    archive, _ = archive_first_and_prior(art("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
                                         art(archived, "E080", "recovered_code5_words.npz"), pn, generator)
    directory = read(art("v089-code5-directory-final", "E091", "code5_directory_checks.json"))
    e092 = read(art("v090-code5-trajectories", "E092", "code5_record_trajectories.json"))
    meta = read(art(archived, "E080", "archive_code5_recovery.json"))["frames"]
    clock = read(art("v077-code5-time-links", "E079", "code5_frame_count.json"))
    epoch = clock["results"]["qin_archive"]["scores"]["dynamic_positive"]["offset"]
    archive_times = {r["archive_index"]: {"counter": r["counter"], "content": r["counter"] - epoch,
                                          "arrival": r["waveform_frame"], "group": r["group"],
                                          "on_waveform_grid": r["on_waveform_grid"]} for r in meta}
    rows, controls = state_rows(archive, directory["archive"], archive_times, directory)
    held_frames = set(e092["held_frames"])
    held = np.array([i for i, r in enumerate(rows) if r["frame"] in held_frames and not r["missing_direct_record_bits"]])
    pairs = group_endpoints(rows, held)
    archive_metrics = {"ranges": magnitude_ranges(rows), "state_rows": rows, "plausible_state_format_counts": controls,
                       "direct_state_frames": sum(not r["missing_direct_record_bits"] for r in rows),
                       "held_group_endpoint_frames": [[rows[i]["frame"], rows[j]["frame"]] for i, j in zip(*pairs)],
                       "held_group_endpoints_content": compare_pairs(rows, *pairs),
                       "held_group_endpoints_arrival": compare_pairs(rows, *pairs, clock="arrival"),
                       "held_group_endpoints_wrong_rate": compare_pairs(rows, *pairs, frame_rate=RATE / 4),
                       "held_adjacent_content": compare_pairs(rows, *adjacent(rows, held, "content"))}
    raw = {}
    for stream, label, path, result, key in (
        ("legacy_480", "480MSps", old_words, art(old, "E012", "code5_parity.json"), "hard_bits"),
        ("ut_250", "250MSps", art("v022-ut-replay", "E014", "ut_observations.npz"),
         art("v022-ut-replay", "E014", "ut_transfer.json"), "hard_code5_words"),
    ):
        words = raw_record_set(path, read(result)["words"], key, pn, generator)
        first = {i: w[0] for i, w in words.items() if 0 in w}
        cr = clock["rows"][stream]
        anchor = cr[0]["counter"] - cr[0]["waveform_frame"]
        timestamps = {r["record_index"]: {"counter": r["counter"], "content": r["counter"] - anchor,
                                           "arrival": r["waveform_frame"]} for r in cr}
        states, wrong_formats = state_rows(first, directory["raw"][label], timestamps, directory)
        idx = np.arange(len(states))
        raw[label] = {"state_rows": states, "ranges": magnitude_ranges(states), "plausible_state_format_counts": wrong_formats,
                      "adjacent_arrival": compare_pairs(states, *adjacent(states, idx, "arrival"), clock="arrival"),
                      "first_to_last_arrival": compare_pairs(states, idx[:1], idx[-1:], clock="arrival")}
    lengths = sorted({r["directory_length"] for r in rows})
    output = {"schema": "starlink-code5-orbit-state/v1", "field_layout": {"position_offsets": [60, 124, 188],
              "velocity_offsets": [252, 284, 316], "format": "<3d3f", "bit_order": "little"},
              "constants": {"mu_m3_s2": GM, "ellipsoid_a_m": EARTH_A, "j2": EARTH_J2,
                            "angular_rate_rad_s": EARTH_OMEGA, "frame_rate_hz": RATE},
              "project_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              "leo_frame_model_sha256": file_hash(EXTERNAL_LEO_ROOT / LEO_FRAMES), "leo_frame_model": LEO_FRAMES.as_posix(),
              "archive": archive_metrics, "raw": raw,
              "window_coverage": coverage(e092["narrowband"], lengths, generator, directory["crc_polynomial_including_top_bit"])}
    target = experiment_output("E094")
    target.mkdir(parents=True, exist_ok=True)
    (target / "code5_orbit_state.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("Qin held endpoints:", archive_metrics["held_group_endpoints_content"])
    for label, capture in raw.items():
        print(label, capture["ranges"], capture["first_to_last_arrival"])


if __name__ == "__main__":
    main()
