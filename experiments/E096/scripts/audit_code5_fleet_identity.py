"""Read the Code 5 fleet number and verify it against ranked historical orbits."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from sgp4.api import Satrec, SatrecArray

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, raw_record_set
from audit_code5_gps_epoch import GPS_UNIX_EPOCH, GPS_UTC_OFFSET, NS, integer_at, split_timestamp
from audit_code5_orbit_state import LEO_FRAMES, file_hash, unpack_state
from audit_code5_record_trajectories import crc_rows, field_information, interval_bits
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


ID_OFFSET = 11
IRREDUCIBLE_BITS = 15
SCAN_BITS = 66 * 8
CATALOG_SOURCE = PROJECT_ROOT / "provenance/code5_identity_orbit_catalogs.json"


def fleet_number(word):
    return integer_at(word, ID_OFFSET, IRREDUCIBLE_BITS)


def name_number(name):
    match = re.fullmatch(r"STARLINK-(\d+)", name)
    if not match:
        raise ValueError(f"Unexpected catalog name: {name}")
    return int(match[1])


def absolute_ns(row):
    return row["gps_ns"] + (GPS_UNIX_EPOCH - GPS_UTC_OFFSET) * NS


def catalogue_rows(path):
    with ZipFile(path) as archive, archive.open("tle_data.csv") as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))


def propagate(frames, satellite, indices, states, epochs, shift=0):
    p = np.array([states[i]["position_m"] for i in indices])
    v = np.array([states[i]["velocity_m_s"] for i in indices])
    t = np.array([absolute_ns(epochs[i]) + shift * NS for i in indices], dtype=np.int64)
    jd, fraction = frames.julian_day_from_utc_ns(t)
    call = satellite.sgp4_array if isinstance(satellite, Satrec) else satellite.sgp4
    err, r, u = call(jd, fraction)
    xyz, vel = frames.teme_to_ecef(r, u, frames.greenwich_mean_sidereal_time_rad(jd, fraction))
    return err, np.linalg.norm(xyz * 1000 - p, axis=-1), np.linalg.norm(vel * 1000 - v, axis=-1)


def record_summary(record, p, v):
    return {"catalog_id": int(record["satellite_id"]), "name": record["satellite_name"],
            "tle_epoch": record["epoch"], "source": record["data_source"],
            "position_m_median": float(np.median(p)), "velocity_m_s_median": float(np.median(v)),
            "position_m_range": [float(np.min(p)), float(np.max(p))],
            "velocity_m_s_range": [float(np.min(v)), float(np.max(v))]}


def orbital_rank(catalogue, frames, states, epochs, indices):
    # Evaluate everyone at the same three anchors before using the full frame set.
    anchors = indices[[0, len(indices) // 2, -1]]
    satellites = [Satrec.twoline2rv(r["tle_line1"], r["tle_line2"]) for r in catalogue]
    if any(sat.satnum != int(row["satellite_id"]) for sat, row in zip(satellites, catalogue, strict=True)):
        raise ValueError("TLE catalog identifier changed during propagation")
    error, p, v = propagate(frames, SatrecArray(satellites), anchors, states, epochs)
    usable = np.flatnonzero(np.all(error == 0, axis=1) & np.all(np.isfinite(p) & np.isfinite(v), axis=1))
    p_order = usable[np.argsort(np.median(p[usable], axis=1), kind="stable")]
    v_order = usable[np.argsort(np.median(v[usable], axis=1), kind="stable")]
    selected = int(p_order[0])
    top = catalogue[selected]
    reevaluation = {}
    for shift in (0, -18, 18):
        errors, pp, vv = propagate(frames, satellites[selected], indices, states, epochs, shift)
        if np.any(errors):
            raise ValueError("Selected TLE failed re-evaluation")
        reevaluation[str(shift)] = record_summary(top, pp, vv)
    return {"records": len(catalogue), "objects": len({r["satellite_id"] for r in catalogue}),
            "usable_records": len(usable), "anchor_frames": [states[i]["frame"] for i in anchors],
            "frame_count_for_reevaluation": len(indices),
            "starlink_objects": len({r["satellite_id"] for r in catalogue if re.fullmatch(r"STARLINK-\d+", r["satellite_name"])}),
            "position_order": [record_summary(catalogue[i], p[i], v[i]) for i in p_order[:10]],
            "velocity_order": [record_summary(catalogue[i], p[i], v[i]) for i in v_order[:10]],
            "selected_full_set": reevaluation}


def nearest_tle_rows(sources):
    result = []
    for source in sources:
        raw = source["response_utf8"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("Nearest-TLE response changed")
        record, = json.loads(raw)[0]["orbital_data"]
        result.append(record)
    return result


def codeword_receipts(words, states, clocks):
    if set(words) != {r["frame"] for r in states} or [r["frame"] for r in states] != [r["frame"] for r in clocks]:
        raise ValueError("Frame membership changed")
    packed, result = [], []
    for state, time in zip(states, clocks, strict=True):
        frame = state["frame"]
        data, seen = words[frame]
        length = data[0]
        record = data[length:]
        raw = int.from_bytes(record, "little")
        position, velocity = unpack_state(record)
        if (length != state["directory_length"] or not np.array_equal(position, state["position_m"])
                or not np.array_equal(velocity, state["velocity_m_s"])
                or split_timestamp(raw) != (time["gps_seconds"], time["fractional_ns"])):
            raise ValueError(f"Codeword at frame {frame} differs from the authenticated state and time")
        start = length * 8 + ID_OFFSET
        direct_count = int(seen[start:start + IRREDUCIBLE_BITS].sum())
        result.append({"frame": frame, "length": length, "prefix_hex": record[:7].hex(),
                       "fleet_number": fleet_number(raw), "direct_id_bits": direct_count})
        packed.append(raw)
    return packed, result


def bit_validation(packed, rows, expected):
    values = [row["fleet_number"] for row in rows]
    compared = np.array(values) == expected
    observed = np.array([r["direct_id_bits"] == IRREDUCIBLE_BITS for r in rows])
    return {"range_of_equivalent_widths": [w for w in range(IRREDUCIBLE_BITS, 35)
                                            if all(integer_at(x, ID_OFFSET, w) == expected for x in packed)],
            "frames": len(values), "matches": int(compared.sum()), "direct_frames": int(observed.sum()),
            "direct_matches": int((observed & compared).sum()),
            "min_width_minus_one_matches": sum(integer_at(w, ID_OFFSET, IRREDUCIBLE_BITS - 1) == expected for w in packed),
            "one_bit_earlier_matches": sum(integer_at(w, ID_OFFSET - 1, IRREDUCIBLE_BITS) == expected for w in packed),
            "one_bit_later_matches": sum(integer_at(w, ID_OFFSET + 1, IRREDUCIBLE_BITS) == expected for w in packed),
            "reversed_order_matches": sum(int(f"{v:015b}"[::-1], 2) == expected for v in values)}


def coverage_limits(windows, lengths, generator, poly):
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    unknown = interval_bits(8, 1024)
    packed = [int.from_bytes(row.tobytes(), "little") & unknown
              for row in np.packbits(code, axis=1, bitorder="little")]
    checks = {length: [c & unknown for c in crc_rows(length, poly)] for length in lengths}
    results = []
    for window in windows:
        positions = window["codeword_coordinates"]
        observations = [packed[i] for i in positions]
        bounds = []
        for length in lengths:
            start = 8 * length + ID_OFFSET
            target = interval_bits(start, start + IRREDUCIBLE_BITS)
            bounds.append({"length": length, "record_bits": [i - 8 * length for i in positions
                                                             if start <= i < start + IRREDUCIBLE_BITS],
                           "rank": field_information(observations, [], target),
                           "rank_with_crc": field_information(observations, checks[length], target)})
        results.append({"kind": window["kind"], "labels": window["labels"], "bounds": bounds})
    return results


def main():
    require_run_context()
    inputs = {}

    def art(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    run = "v010-code5-parity-02"
    legacy = art(run, "E012", "code5_observations.npz")
    with np.load(legacy, allow_pickle=False) as f:
        generator = f["parity_generator"]
    model = read(art("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    archive_words, _ = archive_first_and_prior(art("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
                                               art("v078-archive-code5-links", "E080", "recovered_code5_words.npz"), pn, generator)
    e094 = read(art("v092-code5-orbit-state", "E094", "code5_orbit_state.json"))
    e095 = read(art("v093-code5-gps-epoch", "E095", "code5_gps_epoch.json"))
    e092 = read(art("v090-code5-trajectories", "E092", "code5_record_trajectories.json"))
    e091 = read(art("v089-code5-directory-final", "E091", "code5_directory_checks.json"))
    source = (EXTERNAL_LEO_ROOT / LEO_FRAMES).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != e095["leo_source_sha256"]:
        raise ValueError("LEO frame transform differs from E095")
    frames = ModuleType("leo_frames_e096")
    exec(compile(source, str(LEO_FRAMES), "exec"), frames.__dict__)

    words = {"qin_2025": archive_words}
    states = {"qin_2025": e094["archive"]["state_rows"]}
    clocks = {"qin_2025": e095["archive"]}
    for key, path, receipt, array in (
        ("480MSps", legacy, art(run, "E012", "code5_parity.json"), "hard_bits"),
        ("250MSps", art("v022-ut-replay", "E014", "ut_observations.npz"),
         art("v022-ut-replay", "E014", "ut_transfer.json"), "hard_code5_words"),
    ):
        measured = raw_record_set(path, read(receipt)["words"], array, pn, generator)
        words[key] = {i: w[0] for i, w in measured.items() if 0 in w}
        states[key] = e094["raw"][key]["state_rows"]
        clocks[key] = e095["raw"][key]["rows"]

    inputs[CATALOG_SOURCE] = file_hash(CATALOG_SOURCE)
    catalogues = read(CATALOG_SOURCE)
    comparisons = {}
    held = set(e095["ground_link"]["held_frames"])
    for key, entry in catalogues["snapshots"].items():
        path = PROJECT_ROOT / entry["path"]
        inputs[path] = file_hash(path)
        if inputs[path] != entry["sha256"]:
            raise ValueError("Historical archive ZIP changed")
        catalogue = catalogue_rows(path)
        if len(catalogue) != entry["records"]:
            raise ValueError("Historical archive membership changed")
        indices = np.array([i for i, s in enumerate(states[key]) if key != "qin_2025" or s["frame"] in held])
        comparisons[key] = orbital_rank(catalogue, frames, states[key], clocks[key], indices)
    expected_qin = name_number(comparisons["qin_2025"]["position_order"][0]["name"])
    expected = {"qin_2025": expected_qin, "250MSps": expected_qin}
    expected["480MSps"] = name_number(comparisons["480MSps"]["position_order"][0]["name"])
    legacy_indices = np.arange(len(states["480MSps"]))
    nearest = orbital_rank(nearest_tle_rows(catalogues["legacy_nearest_receipts"]), frames,
                           states["480MSps"], clocks["480MSps"], legacy_indices)
    validations = {}
    for key, packet in words.items():
        packed, records = codeword_receipts(packet, states[key], clocks[key])
        summary = bit_validation(packed, records, expected[key])
        validations[key] = {"predicted_from_catalog": expected[key], "summary": summary, "frames": records}
        if key == "qin_2025":
            cal = set(e092["calibration_frames"])
            discovery = [p for p, r in zip(packed, records, strict=True) if r["frame"] in cal]
            validations[key]["discovery_matching_offsets"] = [offset for offset in range(SCAN_BITS - IRREDUCIBLE_BITS + 1)
                                                               if all(integer_at(w, offset, IRREDUCIBLE_BITS) == expected[key]
                                                                      for w in discovery)]
            held_rows = [i for i, r in enumerate(records) if r["frame"] in held]
            validations[key]["held"] = bit_validation([packed[i] for i in held_rows], [records[i] for i in held_rows], expected[key])

    lengths = sorted({r["directory_length"] for r in states["qin_2025"]})
    bounds = coverage_limits(e092["narrowband"], lengths, generator, e091["crc_polynomial_including_top_bit"])
    report = {"schema": "starlink-code5-fleet-identity/v1", "id_bit_start": ID_OFFSET,
              "validated_low_bits": IRREDUCIBLE_BITS, "leo_source_sha256": digest,
              "inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              "historical_catalogs": comparisons, "legacy_seven_nearest_tles": nearest,
              "field_validation": validations, "ideal_band_coverage": bounds}
    out = experiment_output("E096")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_fleet_identity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for key, item in validations.items():
        print(key, item["predicted_from_catalog"], item["summary"])
    for key, item in comparisons.items():
        print(key, item["records"], item["usable_records"], item["position_order"][:2], item["velocity_order"][0])


if __name__ == "__main__":
    main()
