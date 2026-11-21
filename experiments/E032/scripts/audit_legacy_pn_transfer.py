"""Test the frozen UT payload PN15 model on the older wideband recording."""

from collections import Counter
import json
from pathlib import Path
import sys
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions, matching_run
from audit_ut_repeated_data import bit_weights, checked_artifact, fundamental_period, read, scan_pattern, sort_points
from receiver_models import pn_bits

LEGACY = {
    5: ("v010-code5-parity-02", "E012", "code5_observations", "hard_bits", "code5_parity", 1024),
    9: ("v013-code9-time-unwrapped", "E015", "code9_observations", "hard_bits", "code9_parity", 1280),
    21: ("v014-code21-repeat", "E016", "code21_observations", "hard_bits", "code21", 5120),
    28: ("v019-code28-reproduction-repeat", "E021", "code28_reproduction", "primary_hard_bits", "code28_reproduction", 8960),
    37: ("v015-code37-verified", "E017", "code37_observations", "hard_bits", "code37", 4736),
    68: ("v021-code68-erasure-audit", "E023", "code68_observations", "gray_bits", "code68_audit", None),
}
SEED = 2026112132
LAG = 24
HISTORY = 15 + LAG
MIN_LENGTH = 144
C68_BOUNDARY = 6144


def strongest(differences):
    best = matching_run(differences)
    best["accepted_144"] = best["length"] >= MIN_LENGTH and 10 * best["mismatches"] <= best["length"]
    return best


def longest_gain(scores):
    prefix = np.r_[0., np.cumsum(scores)]
    gains = prefix - np.minimum.accumulate(prefix)
    stop = int(np.argmax(gains))
    start = int(np.argmin(prefix[:stop + 1]))
    return start, stop, float(gains[stop])


def recurrence_residual(bits, tap):
    check = bits[..., 15:] ^ bits[..., 15-tap:-tap] ^ bits[..., :-15]
    return check[..., LAG:] ^ check[..., :-LAG]


def canonical_byte_template(data, start, stop):
    coords = np.arange(start, stop) % LAG
    counts = np.bincount(coords, minlength=LAG)
    ones = np.bincount(coords, weights=data[start:stop], minlength=LAG)
    template = np.packbits(2 * ones > counts).tobytes()
    return min(template[shift:] + template[:shift] for shift in range(len(template)))


def compare_templates(reference, motif):
    i = np.arange(len(reference)) % len(motif)
    return {
        "reference_period": fundamental_period(reference),
        "candidate_period": fundamental_period(motif),
        "length": len(reference),
        "min_cyclic_reverse_or_polarity_distance": min(
            int(np.count_nonzero(reference != (np.roll(m, shift)[i] ^ polarity)))
            for m in (motif, motif[::-1]) for shift in range(len(motif)) for polarity in (0, 1)
        ),
    }


def systematic_regions(bits, rows, p, pn_controls, motif12):
    for row, word in zip(rows, bits, strict=True):
        fixed = strongest(word ^ p)
        controls = {name: strongest(word ^ v) for name, v in pn_controls.items()}
        row["fixed_zero_source"] = fixed
        row["controls"] = controls
        row["e031_template"] = scan_pattern(word ^ p, motif12)


def periodic_regions(bits, rows, p):
    natural = np.arange(bits.shape[1]) ^ 7
    a = bits[:, natural]
    dewhitened = (bits ^ p)[:, natural]
    traces = recurrence_residual(a, 14)
    controls = (recurrence_residual(bits, 14), recurrence_residual(a, 13))
    for idx, row in enumerate(rows):
        check = strongest(traces[idx])
        row["lag24"] = check
        row["lag24_controls"] = [strongest(control[idx]) for control in controls]
        if check["accepted_144"]:
            lo, hi = check["start"], check["stop"] + HISTORY
            row["lag24_source_support"] = [lo, hi]
    groups = np.unique([row["frame"] for row in rows])
    fit_frames = set(np.random.default_rng(SEED).permutation(groups)[:len(groups)//2].tolist())
    templates = Counter()
    for row, word in zip(rows, dewhitened, strict=True):
        if row["frame"] not in fit_frames or "lag24_source_support" not in row:
            continue
        lo, hi = row["lag24_source_support"]
        motif = canonical_byte_template(word, lo, hi)
        if motif not in (bytes(3), b"\xff" * 3):
            templates[motif] += 1
    for row, word in zip(rows, dewhitened, strict=True):
        row["periodic_partition"] = "calibration" if row["frame"] in fit_frames else "reserved"
        row["frozen_nonconstant_templates"] = {
            motif.hex(): scan_pattern(word, np.unpackbits(np.frombuffer(motif, np.uint8)))
            for motif in templates
        }
    return {"calibration_frames": sorted(fit_frames), "reserved_frames": sorted(set(groups.tolist()) - fit_frames),
            "templates_natural_order": {motif.hex(): count for motif, count in templates.items()}}


def code68_waveform(archive, rows, expected, pn_controls):
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    points = (levels[:, None] + 1j * levels[None, :]).ravel()
    lut = sort_points(points, archive["frozen_gray_mapping"])
    predictions = {"fixed": expected, **pn_controls}
    prediction_labels = {name: v.reshape(-1, 4) @ bit_weights(4) for name, v in predictions.items()}
    observed = archive["symbols"]
    costs = abs(observed[:, :, None] - lut[None, None, :]) ** 2
    for j, row in enumerate(rows):
        fixed = row["fixed_zero_source"]
        left = (fixed["start"] + 3) // 4
        stop = C68_BOUNDARY // 4
        nearest = costs[j].min(axis=1)
        aligned = {name: costs[j, np.arange(observed.shape[1]), labels] for name, labels in prediction_labels.items()}
        row["soft_change_points"] = {}
        for scale in (.02, .03, .04):
            base = logsumexp(-costs[j] / scale, axis=1) - np.log(16)
            a, b, gain = longest_gain(-aligned["fixed"] / scale - base)
            row["soft_change_points"][str(scale)] = {"start_bit": 4 * a, "stop_bit": 4 * b, "gain": gain}
        if fixed["accepted_144"]:
            n = (stop - left)
            if n <= 0:
                raise ValueError("Detected PN interval starts at or after the candidate boundary")
            row["waveform_to_candidate_boundary"] = {
                "start_bit": 4 * left, "stop_bit": C68_BOUNDARY, "symbols": n,
                "mean_squared_distances": {name: float(v[left:stop].mean()) for name, v in aligned.items()},
                "nearest_point_mse": float(nearest[left:stop].mean()),
                "edge_window_bits": 256,
                "before_edge_bit_errors": int(np.count_nonzero(
                    archive["gray_bits"][j, C68_BOUNDARY-256:C68_BOUNDARY]
                    != expected[C68_BOUNDARY-256:C68_BOUNDARY])),
                "after_edge_bit_errors": int(np.count_nonzero(
                    archive["gray_bits"][j, C68_BOUNDARY:C68_BOUNDARY+256]
                    != expected[C68_BOUNDARY:C68_BOUNDARY+256])),
            }


def word_metadata(records):
    rows = []
    for j, (row, following) in enumerate(zip(records, [*records[1:], {}], strict=True)):
        final_word = row["frame"] != following.get("frame") or row.get("field") != following.get("field")
        rows.append({"frame": row["frame"], "word": row["word"], "start_resource": row["start_resource"],
                     "candidate": j, "allocation_final_word": final_word})
    return rows


def main():
    require_run_context()
    inputs = {}
    previous = checked_artifact("v029-ut-repeated-data-waveform", "E031", "ut_repeated_data.json", inputs)
    tail = checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs)
    model_record = read(previous)
    fit = model_record["pn_model"]
    motif = np.unpackbits(np.frombuffer(bytes.fromhex(model_record["template_absolute_phase_lsb_first_hex"]), np.uint8),
                          bitorder="little")
    tail_folds = read(tail)["cycle_folds"]
    sequence = pn_bits()
    formats = {}
    for code, (run, experiment, obs_name, key, report_name, known_stop) in LEGACY.items():
        archive = checked_artifact(run, experiment, obs_name + ".npz", inputs)
        meta = read(checked_artifact(run, experiment, report_name + ".json", inputs))
        rows = word_metadata(meta["word_records" if code == 68 else "words"])
        with np.load(archive, allow_pickle=False) as z:
            labels = z[key]
            bits = labels[:, :labels.shape[1] // 8 * 8]
            positions = np.arange(bits.shape[1])
            fixed = equality_predictions(positions, sequence, fit)
            controls = {
                "phase_minus1": equality_predictions(positions, sequence, {**fit, "phase": fit["phase"] - 1}),
                "phase_plus1": equality_predictions(positions, sequence, {**fit, "phase": fit["phase"] + 1}),
                "unreversed": equality_predictions(positions, sequence, {**fit, "within_unit_reverse": 1}),
            }
            systematic_regions(bits, rows, fixed, controls, motif)
            periodic = periodic_regions(bits, rows, fixed) if code in (21, 28) else None
            if code == 68:
                code68_waveform(z, rows, fixed, controls)
        formats[str(code)] = {"known_systematic_bits": known_stop, "observed_label_bits": labels.shape[1],
                              "searched_bits": bits.shape[1], "words": rows, "periodic": periodic}
        print(code, len(rows), "words;", sum(row["fixed_zero_source"]["accepted_144"] for row in rows),
              "fixed PN runs", flush=True)
    out = experiment_output("E032")
    out.mkdir(parents=True, exist_ok=True)
    templates = {"E031_MCS66": motif[np.arange(len(motif)) ^ 7]}
    for code in ("21", "28"):
        for template in formats[code]["periodic"]["templates_natural_order"]:
            templates[f"Code{code}_{template}"] = np.unpackbits(np.frombuffer(bytes.fromhex(template), np.uint8))
    report = {
        "schema": "starlink-legacy-pn-transfer/v1",
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "pn_model": fit, "periodic_group_seed": SEED, "shared_threshold_bits": MIN_LENGTH,
        "code68_candidate_boundary_bits": C68_BOUNDARY,
        "e031_vs_e025": [compare_templates(np.array(list(fold["binary_template"]), dtype=np.uint8), motif)
                          for fold in tail_folds],
        "byte_natural_template_comparisons": [
            {"left": left, "right": right, **compare_templates(templates[left], templates[right])}
            for left, right in combinations(templates, 2)
        ],
        "formats": formats,
        "scope": "Frozen transfer across two recorded IQ captures. Interval choices and the Code 68 nominal boundary are retrospective. "
                 "PN agreement tests a candidate information region, without identifying the parity encoder or meanings.",
    }
    (out / "legacy_pn_transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
