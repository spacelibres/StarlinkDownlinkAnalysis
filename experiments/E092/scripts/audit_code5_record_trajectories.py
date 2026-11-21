"""Test time-varying Code 5 record bits and narrowband CRC observability."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, raw_record_set, remainder
from audit_code5_frame_count import RELATIVE_BIT, descramble
from audit_h2_projection import TRACK, WAV_MODEL, band_masks, visible_positions
from audit_qam_pn15 import equality_predictions
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


SEED = 2026112193
WIDTH = 14
RECORD_BYTES = 66
TIME_CENTER = 5000.
TIME_SCALE = 3000.
CONTROLS = 199


def sliding_values(bytes_array, width):
    bits = np.unpackbits(bytes_array, axis=1, bitorder="little")
    return np.lib.stride_tricks.sliding_window_view(bits, width, axis=1) @ (1 << np.arange(width))


def interval_bits(low, high):
    return ((1 << (high - low)) - 1) << low


def packed_rank(rows):
    pivots = {}
    for row in rows:
        while row:
            high = row.bit_length() - 1
            if high not in pivots:
                pivots[high] = row
                break
            row ^= pivots[high]
    return len(pivots)


def field_information(measurements, constraints, field):
    rows = list(measurements) + list(constraints)
    remove = ~field
    return (packed_rank(rows) - packed_rank(row & remove for row in rows)
            - packed_rank(constraints) + packed_rank(row & remove for row in constraints))


def crc_rows(length, poly):
    responses = [remainder(1 << i, poly) for i in range(8 * length - 1, -1, -1)]
    return [sum(1 << i for i, v in enumerate(responses) if (v >> j) & 1) for j in range(24)]


def observability(generator, poly, lengths, positions, windows):
    code = np.r_[np.eye(1024, dtype=np.uint8), generator.T]
    unknown = interval_bits(8, 1024)
    code = [int.from_bytes(v.tobytes(), "little") & unknown for v in np.packbits(code, axis=1, bitorder="little")]
    constraints = {L: [r & unknown for r in crc_rows(L, poly)] for L in lengths}
    results = []
    for source in windows:
        hx = [code[i] for i in source["codeword_coordinates"]]
        rows = []
        for L, crc in constraints.items():
            base = L * 8
            targets = {
                "directory": interval_bits(8, base),
                "first_record_span12": interval_bits(44, 56),
                "frame_count14": interval_bits(base + RELATIVE_BIT, base + RELATIVE_BIT + 14),
                "trajectory_bits": sum(interval_bits(base + j, base + j + WIDTH) for j in positions),
            }
            rows.append({"length": L, "targets": {
                name: {"without_crc": field_information(hx, (), mask),
                       "with_crc": field_information(hx, crc, mask)} for name, mask in targets.items()}})
        results.append({**source, "lengths": rows})
    return results


def prefix_bank(words):
    prefixes = [np.frombuffer(word[word[0]:word[0] + RECORD_BYTES], dtype=np.uint8) for word in words]
    return sliding_values(np.stack(prefixes), WIDTH)


def time_model(n, degree=2):
    return np.polynomial.polynomial.polyvander((n - TIME_CENTER) / TIME_SCALE, degree)


def error_summary(gap):
    delta = np.abs(gap)
    return {"count": len(delta), "rmse": float(np.sqrt(np.mean(delta**2))),
            "median_abs": float(np.median(delta)), "max_abs": float(delta.max()),
            "within_one_unit": int((delta <= 1).sum()), "round_exact": int((delta < .5).sum())}


def nominate(values, clock, training):
    design = time_model(clock)
    coeff = np.linalg.lstsq(design[training], values[training], rcond=None)[0]
    errors = abs(values[training] - design[training] @ coeff)
    visible = np.arange(values.shape[1])
    # Exclude every window overlapping the existing 32-bit frame-count field.
    admissible = (visible + WIDTH <= RELATIVE_BIT) | (visible >= RELATIVE_BIT + 32)
    retained = np.flatnonzero(admissible & (np.ptp(values[training], axis=0) >= 64)
                              & (np.quantile(errors, .99, axis=0) < 1.10))
    families = [v.tolist() for v in np.split(retained, np.flatnonzero(np.diff(retained) != 1) + 1) if len(v) >= 2]
    positions = [family[0] for family in families]
    return positions, families, coeff[:, positions]


def wrong_epochs(n, groups, rng):
    cohorts, inverse = np.unique(groups, return_inverse=True)
    centers = np.array([n[groups == g].mean() for g in cohorts])
    for _ in range(CONTROLS):
        shuffled = centers[rng.permutation(len(cohorts))]
        yield n + (shuffled - centers)[inverse]


def archive_curves(words, meta, positions, coeff, counter_offset, rng):
    codes = prefix_bank(words)[:, positions]
    n = np.array([row["waveform_frame"] for row in meta])
    groups = np.array([row["group"] for row in meta])
    on_grid = np.array([row["on_waveform_grid"] for row in meta])
    fit = np.array([row["calibration"] for row in meta])
    counter_time = np.array([row["counter"] - counter_offset for row in meta])
    held = on_grid & ~fit
    selected = time_model(n) @ coeff
    from_count = time_model(counter_time) @ coeff
    ho = codes[held] - selected[held]
    null = np.array([np.median(abs(codes[held] - time_model(t) @ coeff), axis=0)
                     for t in wrong_epochs(n[held], groups[held], rng)])
    linear = time_model(n, 1)
    beta = np.linalg.lstsq(linear[on_grid & fit], codes[on_grid & fit], rcond=None)[0]
    prior = linear @ beta
    records = []
    for j, pos in enumerate(positions):
        records.append({
            "offset": pos, "coefficients": coeff[:, j].tolist(), "held": error_summary(ho[:, j]),
            "linear_held": error_summary((codes - prior)[held, j]),
            "held_groups": len(set(groups[held])),
            "wrong_time_min_median_abs": float(null[:, j].min()),
            "wrong_time_controls_at_least_as_good": int((null[:, j] <= np.median(abs(ho[:, j]))).sum()),
            "off_grid_by_arrival": error_summary((codes - selected)[~on_grid, j]),
            "off_grid_by_payload_count": error_summary((codes - from_count)[~on_grid, j]),
            "frame956_arrival_and_counter_abs_error": [float(v) for v in
                (abs(codes[956, j] - selected[956, j]), abs(codes[956, j] - from_count[956, j]))],
        })
    return records


def raw_curves(raw, time_receipt, positions, rng):
    n = np.array([r["waveform_frame"] for r in time_receipt])
    ids = [r["record_index"] for r in time_receipt]
    values = prefix_bank([raw[i][0][0] for i in ids])[:, positions]
    pairs = np.arange(len(ids)) // 2
    groups = rng.permutation(np.unique(pairs))
    fit = np.isin(pairs, groups[:len(groups) // 2])
    x = np.c_[np.ones(len(ids)), n]
    beta = np.linalg.lstsq(x[fit], values[fit], rcond=None)[0]
    errs = (values - x @ beta)[~fit]
    return {"ids": ids, "training_ids": [i for i, use in zip(ids, fit, strict=True) if use], "clock_frames": n.tolist(),
            "fields": [{"offset": pos, "slope": float(beta[1, j]), "range": int(np.ptp(values[:, j])),
                        "held": error_summary(errs[:, j])} for j, pos in enumerate(positions)]}


def body_checks(first, second, directory, fit, raw):
    starts = [r["directory_length"] + r["decoded_record_span"] for r in directory]
    sample = [a + b[0] for a, b in zip(first, second, strict=True)]
    prefix = Counter(s[e:e + 20] for s, e, take in zip(sample, starts, fit, strict=True) if take).most_common(1)[0][0]
    body = {}
    for label, cohort in [("archive_calibration", np.flatnonzero(fit)), ("archive_held", np.flatnonzero(~fit))]:
        body[label] = {"samples": len(cohort),
                       "second_prefix": sum(sample[i][starts[i]:starts[i] + 20] == prefix for i in cohort),
                       "zero_tail_25": sum(sample[i][starts[i] - 26:starts[i]] == b"\x04" + bytes(25) for i in cohort)}
    for label, words in raw.items():
        checks = []
        for frame in words.values():
            a = frame[0][0]
            b = a + frame[1][0]
            start = a[0] + (int.from_bytes(a[5:7], "little") >> 4)
            checks.append((b[start:start + 20] == prefix, b[start - 26:start] == b"\x04" + bytes(25)))
        body[label] = {"samples": len(checks), "second_prefix": sum(x[0] for x in checks),
                       "zero_tail_25": sum(x[1] for x in checks)}
    return {"archived_prefix_hex": prefix.hex(), "runs": body}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    e091 = "v089-code5-directory-final"
    directory = read(artifact(e091, "E091", "code5_directory_checks.json"))
    assert all(r["crc_pass"] for r in directory["archive"])
    base = "v010-code5-parity-02"
    old_file = artifact(base, "E012", "code5_observations.npz")
    with np.load(old_file, allow_pickle=False) as z:
        generator = z["parity_generator"]
    model = read(artifact("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    e080 = "v078-archive-code5-links"
    data, second = archive_first_and_prior(
        artifact("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
        artifact(e080, "E080", "recovered_code5_words.npz"), pn, generator
    )
    with np.load(artifact(e091, "E091", "code5_directory_following_words.npz"), allow_pickle=False) as z:
        for i, w, m in zip(z["frame_index"], z["bits"], z["observed"], strict=True):
            second[i] = (descramble(np.unpackbits(w), pn), np.unpackbits(m).astype(bool)[:1024])
    timings = read(artifact(e080, "E080", "archive_code5_recovery.json"))["frames"]
    time_control = read(artifact("v077-code5-time-links", "E079", "code5_frame_count.json"))
    counter_offset = time_control["results"]["qin_archive"]["scores"]["dynamic_positive"]["offset"]
    raw_same = "v022-ut-replay"
    raw = {
        "legacy_480": raw_record_set(old_file, read(artifact(base, "E012", "code5_parity.json"))["words"],
                                     "hard_bits", pn, generator),
        "ut_250": raw_record_set(artifact(raw_same, "E014", "ut_observations.npz"),
                                 read(artifact(raw_same, "E014", "ut_transfer.json"))["words"],
                                 "hard_code5_words", pn, generator),
    }
    first = [data[i][0] for i in range(len(data))]
    fit = np.array([r["calibration"] for r in timings])
    clock = np.array([r["waveform_frame"] for r in timings])
    in_grid = np.array([r["on_waveform_grid"] for r in timings])
    candidate, families, coeff = nominate(prefix_bank(first), clock, fit & in_grid)
    rng = np.random.default_rng(SEED)
    curves = archive_curves(first, timings, candidate, coeff, counter_offset, rng)
    existing = read(artifact("v077-leo-code5-projection", "E079", "leo_code5_projection.json"))["windows"]
    inputs.update({ROOT / f: sha256(ROOT / f) for f in (TRACK, WAV_MODEL)})
    bands, _ = band_masks()
    wav = {"kind": "wav", "labels": ["original WAV"],
           "codeword_coordinates": list(visible_positions(228, 2048, bands["wav"]))}
    record = {"schema": "starlink-code5-record-trajectories/v1", "seed": SEED, "searched_width": WIDTH,
              "calibration_frames": np.flatnonzero(fit & in_grid).tolist(), "held_frames": np.flatnonzero(~fit & in_grid).tolist(),
              "families": families, "archive_trajectories": curves, "group_wrong_time_trials": CONTROLS,
              "raw_trajectories": {k: raw_curves(v, time_control["rows"][k], candidate, rng) for k, v in raw.items()},
              "record_boundary_structure": body_checks(first, [second[i] for i in range(len(first))],
                                                       directory["archive"], fit, raw),
              "narrowband": observability(generator, directory["crc_polynomial_including_top_bit"],
                                           sorted({x["directory_length"] for x in directory["archive"]}),
                                           candidate, [wav, *existing]),
              "input_sha256": {p.relative_to(ROOT).as_posix(): v for p, v in inputs.items()}}
    out = experiment_output("E092")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_record_trajectories.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Trajectory starts", candidate, "held frames", len(record["held_frames"]), flush=True)


if __name__ == "__main__":
    main()
