"""Locate the UT binary tail without using any second-header allocation lengths."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, legacy_artifact, require_run_context
from code5_core import LOADED, equalize_grid, reference_gain
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from ut_receiver_core import capture_grid, read_ci16, resample_480

SOURCE = "data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin"
RUN = ROOT / "runs/v022-ut-replay/artifacts/experiments/E014/results"
SEARCH = 65536
SCALES = (.03, .06, .12)
EDGE_PAD = 2
SEED = 2026112125

AXIS = np.array([-3, -1, 1, 3])
QAM16 = (AXIS[:, None] + 1j * AXIS[None, :]).ravel() / np.sqrt(10)
CROSS_AXIS = np.array([-5, -3, -1, 1, 3, 5])
I, Q = np.meshgrid(CROSS_AXIS, CROSS_AXIS)
CROSS32 = (I + 1j * Q)[(abs(I) < 5) | (abs(Q) < 5)] / np.sqrt(20)
BPSK = np.array([-1., 1.])


def cost(values, points, scale):
    squared = abs(values[:, None] - points[None, :]) ** 2
    return -scale * (logsumexp(-squared / scale, axis=1) - np.log(len(points)))


def constellation_costs(values, scale):
    """Equal-prior finite alphabet cost, in normalized mean-power units."""
    binary = cost(values, BPSK, scale)
    nonbinary = np.minimum(cost(values, QAM16, scale), cost(values, CROSS32, scale))
    return binary, nonbinary


def binary_tail(delta, min_side=128):
    """One transition from nonbinary to binary, with both homogeneous fits as controls."""
    if len(delta) < min_side * 2:
        raise ValueError("Too few resources for two sides")
    cumulative = np.r_[0., np.cumsum(delta, dtype=np.float64)]
    k = int(min_side + np.argmax(cumulative[min_side:len(delta) - min_side + 1]))
    margin = float(cumulative[k] - max(0., cumulative[-1]))
    before = float(np.mean(delta[k - min_side:k]))
    after = float(np.mean(delta[k:k + min_side]))
    edge = k if margin > 0 and before > .02 and after < -.02 else None
    return {
        "first_binary_resource": edge,
        "best_split": k,
        "gain_over_homogeneous": margin,
        "mean_delta_before": before,
        "mean_delta_after": after,
    }


def fit_carrier_gain(grid, words, decoded):
    count = len(NONPILOT)
    samples, indices = [], []
    for row, signs in zip(words, decoded, strict=True):
        idx = row["start_resource"] + np.arange(len(signs))
        samples.append(grid.ravel()[idx] * (1 - 2 * signs.astype(float)))
        indices.append(idx % count)
    samples = np.concatenate(samples)
    carrier = np.concatenate(indices)
    counts = np.bincount(carrier, minlength=count)
    gains = (np.bincount(carrier, weights=samples.real, minlength=count)
             + 1j * np.bincount(carrier, weights=samples.imag, minlength=count)) / counts
    if np.min(abs(gains)) < .1:
        raise ValueError("Insufficient Code 5 response")
    return gains


def boundary(values, scale, first):
    a, b = constellation_costs(values, scale)
    r = binary_tail(a - b)
    for key in ("first_binary_resource", "best_split"):
        if r[key] is not None:
            r[key] += first
    return r


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_samples(source, reference):
    if sha256(source) != reference["source_sha256"]:
        raise ValueError("Raw capture changed")
    cache = ROOT / ".cache/ut250_c69e820_480_6145.npy"
    if cache.is_file() and sha256(cache) == reference["resampling"]["sha256"]:
        return np.load(cache, allow_pickle=False, mmap_mode="r")
    return resample_480(read_ci16(source))


def analyze_frame(samples, case, sss, words, decoded):
    spectra, slots = capture_grid(samples, case["pss"]["epoch"], case["sampling_scale"], case["cfo_hz"])
    equal = equalize_grid(spectra[1:], spectra[0], sss, slots[1:])
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)])[:48].astype(float)
    grid = equal["nonpilot"]
    grid = grid / reference_gain(grid.ravel()[:48], signs)
    fine = grid / fit_carrier_gain(grid, words, decoded)[None, :]
    first = grid.size - SEARCH
    patch = fine.ravel()[first:]
    rotor = np.column_stack([pn_reference(int(k), slots[1:]) for k in NONPILOT]).ravel()[first:]
    result = {
        "candidate": case["candidate_index"],
        "search_start_resource": first,
        "frame_end_resource": grid.size,
        "scales": {str(scale): boundary(patch, scale, first) for scale in SCALES},
        "pilot_only": boundary(grid.ravel()[first:], .06, first),
        "retaining_pn": boundary(patch * rotor, .06, first),
    }
    start = result["scales"]["0.06"]["first_binary_resource"]
    if start is None:
        return result, {}
    tail = np.asarray(fine.ravel()[start:])
    signs = tail.real < 0
    occupancy = Counter(signs.tolist())
    result.update(tail_resources=len(tail), negative_symbols=int(occupancy[True]),
                  positive_symbols=int(occupancy[False]), nearest_binary_mse=float(np.mean(abs(tail - (1 - 2 * signs.astype(float))) ** 2)),
                  constant_positive_mse=float(np.mean(abs(tail - 1) ** 2)),
                  constant_negative_mse=float(np.mean(abs(tail + 1) ** 2)))
    return result, {
        "symbols": tail.astype(np.complex64),
        "signs": signs,
        "preceding_symbols": fine.ravel()[start - EDGE_PAD:start].astype(np.complex64),
    }


def minimum_period(sequences, upper=256):
    """First lag with no within-frame bit discrepancies."""
    for period in range(1, upper + 1):
        if all(len(bits) >= 2 * period and np.array_equal(bits[:-period], bits[period:])
               for bits in sequences):
            return period
    raise ValueError("No exact period within the tested bank")


def cycle_template(sequences, period):
    values = np.stack([bits[:period] for bits in sequences])
    return values.sum(axis=0) > len(values) / 2


def score_template(template, symbols, absolute_start):
    """Choose the local start using 120 signs, then predict disjoint resources."""
    decisions = symbols.real < 0
    offsets = np.arange(-EDGE_PAD, EDGE_PAD + 1)
    position = np.arange(len(symbols)) - EDGE_PAD
    phases = (position[None, :] - offsets[:, None]) % len(template)
    predicted = template[phases]
    fit = (position >= EDGE_PAD) & (position < EDGE_PAD + 120)
    test = position >= EDGE_PAD + 120
    fit_losses = np.count_nonzero(predicted[:, fit] != decisions[None, fit], axis=1)
    best = min(range(len(offsets)), key=lambda i: (fit_losses[i], abs(offsets[i]), offsets[i]))
    offset = int(offsets[best])
    at_slot, q = divmod(absolute_start + position[test], len(NONPILOT))
    start_slot, start_q = divmod(absolute_start + offset, len(NONPILOT))
    reset_index = (q - start_q) % len(template)
    reverse_stride = (q - start_q + 16 * (at_slot - start_slot)) % len(template)
    return {
        "edge_adjustment": offset,
        "edge_fit_disagreements": int(fit_losses[best]),
        "edge_fit_ties": int(np.count_nonzero(fit_losses == fit_losses[best])),
        "held_resources": int(test.sum()),
        "held_disagreements": int(np.count_nonzero(predicted[best, test] != decisions[test])),
        "held_disagreements_without_adjustment": int(np.count_nonzero(predicted[EDGE_PAD, test] != decisions[test])),
        "held_disagreements_zero_slot_stride": int(np.count_nonzero(template[reset_index] != decisions[test])),
        "held_disagreements_opposite_slot_stride": int(np.count_nonzero(template[reverse_stride] != decisions[test])),
    }


def validate_cycle(rows, arrays):
    ids = sorted(row["candidate"] for row in rows)
    splits = np.random.default_rng(SEED).permutation(ids)
    starts = {row["candidate"]: row["scales"]["0.06"]["first_binary_resource"] for row in rows}
    observations = {i: np.r_[arrays[f"frame_{i}_preceding_symbols"], arrays[f"frame_{i}_symbols"]] for i in ids}
    records = []
    rng = np.random.default_rng(SEED + 1)
    for train, held in ((splits[:4], splits[4:]), (splits[4:], splits[:4])):
        train_signs = [arrays[f"frame_{i}_signs"] for i in train]
        period = minimum_period(train_signs)
        template = cycle_template(train_signs, period)
        tests = [{"candidate": int(i), **score_template(template, observations[i], starts[i])} for i in sorted(held)]
        control_errors = []
        for _ in range(399):
            shuffled = rng.permutation(template)
            control_errors.append(sum(score_template(shuffled, observations[i], starts[i])["held_disagreements"] for i in held))
        records.append({
            "calibration_frames": sorted(train.tolist()),
            "reserved_frames": sorted(held.tolist()),
            "period": period,
            "binary_template": "".join(map(str, template.astype(int))),
            "signed_template": [int(1 - 2 * bit) for bit in template],
            "reserved_frame_tests": tests,
            "same_composition_permutation_errors": control_errors,
        })
    return records


def audit_lengths(rows, cases, cycle_results):
    # First five lengths are anchored by the public example. Other codes are a finite shared-length extension.
    units = {57: 2048, 66: 4096, 70: 4096, 71: 3277, 78: 3277, 85: 3277, 88: 3277, 92: 3277}
    cases = {case["candidate_index"]: case for case in cases}
    offsets = {r["candidate"]: r["edge_adjustment"] for fold in cycle_results for r in fold["reserved_frame_tests"]}
    comparisons = []
    for row in rows:
        frame = row["candidate"]
        case = cases[frame]
        header = case["h2"]
        payload_start = case["predicted_h2_start"] + 114 * header["blocks"]
        entries = header["layout20"]["entries"]
        endpoint = payload_start + sum(units[e["mcs_candidate"]] * e["quantity"] for e in entries)
        joint_end = payload_start
        for entry in entries:
            unit, quantity = units[entry["mcs_candidate"]], entry["quantity"]
            joint_end += (16384 * quantity + 4) // 5 if unit == 3277 else unit * quantity
        observed = row["scales"]["0.06"]["first_binary_resource"]
        refined = observed + offsets[frame]
        comparisons.append({
            "candidate": frame, "conditional_predicted_endpoint": endpoint,
            "candidate_joint_rounding_endpoint": joint_end,
            "modulation_endpoint_minus_prediction": observed - endpoint,
            "template_refined_endpoint": refined,
            "template_refined_endpoint_minus_prediction": refined - endpoint,
            "template_refined_endpoint_minus_joint_rounding": refined - joint_end,
        })
    return {"candidate_resource_units": units, "comparisons": comparisons}


def main():
    require_run_context()
    source = ROOT / SOURCE
    report_path, words_path = [RUN / path for path in ("ut_transfer.json", "ut_observations.npz")]
    reference = json.loads(report_path.read_text(encoding="utf-8"))
    samples = load_samples(source, reference)
    sss_file = legacy_artifact("reference/qin/sssVec.mat")
    sss = loadmat(sss_file)["sssVecFull"].ravel()[LOADED % 1024]
    rows, arrays = [], {}
    with np.load(words_path, allow_pickle=False) as saved:
        decoded = saved["decoded_code5_words"]
    for case in reference["candidates"]:
        if not case["complete_frame"]:
            continue
        candidate = case["candidate_index"]
        indices = [i for i, word in enumerate(reference["words"])
                   if word["candidate_index"] == candidate and word["initial_parity_disagreements"] == 0]
        words = [reference["words"][i] for i in indices]
        record, evidence = analyze_frame(samples, case, sss, words, decoded[indices])
        rows.append(record)
        arrays.update({f"frame_{candidate}_{name}": value for name, value in evidence.items()})
        print(candidate, record["scales"]["0.06"]["first_binary_resource"],
              record.get("tail_resources"), record["pilot_only"]["first_binary_resource"], flush=True)
    published = 299288
    cycle_results = validate_cycle(rows, arrays)
    result = {
        "schema": "starlink-ut-binary-tail/v1",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                          for p in [source, report_path, words_path, sss_file]},
        "signal_results": rows,
        "excluded_incomplete_frame_ids": [c["candidate_index"] for c in reference["candidates"] if not c["complete_frame"]],
        "published_f250_exclusive_payload_end": published,
        "f250_error_resources": rows[0]["scales"]["0.06"]["first_binary_resource"] - published
        if "tail_resources" in rows[0] else None,
        "cycle_folds": cycle_results,
        "conditional_length_audit": audit_lengths(rows, reference["candidates"], cycle_results),
        "cycle_group_seed": SEED,
        "permutation_seed": SEED + 1,
        "scope": "Conditional finite modulation bank and same-capture receiver; payload boundaries and bit semantics are separate tests.",
    }
    out = experiment_output("E025")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ut_binary_tail.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "ut_binary_tail.npz", **arrays)


if __name__ == "__main__":
    main()
