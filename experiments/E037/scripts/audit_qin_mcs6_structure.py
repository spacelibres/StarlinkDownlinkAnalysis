"""Evaluate the Code 6 record hypothesis and audit PN15 mismatches by carrier."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from receiver_models import pn_bits


K = 10240
PREFIX = bytes.fromhex("e0c000")
RECORD_WIDTH = {0xC: 5, 0xE: 2}
MIN_C_RECORDS = 5
SCAN_LIMIT = 256
GUARD = 32
QUARTERS = np.array([1, 1j, -1, -1j])


def record_candidate(source):
    """Offsets refer to MSB-first bytes at the receiver-ordered word origin."""
    prefix_errors = sum((left ^ right).bit_count() for left, right in zip(source[2:5], PREFIX, strict=True))
    start = 5
    limit = min(len(source), SCAN_LIMIT)
    records = []
    while start < limit:
        tag = source[start] >> 4
        width = RECORD_WIDTH.get(tag)
        if width is None or start + width > limit:
            break
        record = {"start": start, "tag": tag}
        if tag == 0xC:
            record["low_three_bits"] = source[start + width - 1] & 7
        records.append(record)
        start += width
    count = Counter(record["tag"] for record in records)
    return {"prefix_bit_errors": prefix_errors, "records": records, "c_records": count[0xC],
            "e_records": count[0xE], "stop_byte": start,
            "stop_tag": source[start] >> 4 if start < limit else None,
            "match": prefix_errors == 0 and count[0xC] >= MIN_C_RECORDS}


def grammar_controls(hard, expected, other_phases):
    plaintext = np.packbits(hard ^ expected, bitorder="big").tobytes()
    return {
        "fixed": record_candidate(plaintext),
        "without_pn": record_candidate(np.packbits(hard, bitorder="big").tobytes()),
        "byte_early": record_candidate(bytes(1) + plaintext),
        "byte_late": record_candidate(plaintext[1:]),
        **{name: record_candidate(np.packbits(hard ^ pn, bitorder="big").tobytes())
           for name, pn in other_phases.items()},
    }


def padding_observations(data, rows, pn):
    cases = []
    words = data["mcs6_symbols"]
    if len(rows) != len(words):
        raise ValueError("E036 word metadata no longer matches its observations")
    for word, (row, observed) in enumerate(zip(rows, words, strict=True)):
        if row["frame"] != int(data["mcs6_frames"][word]):
            raise ValueError("E036 frame records no longer align")
        if not row["fixed"]["accepted"]:
            continue
        start = GUARD + row["fixed"]["start"]
        stop = min(K, row["fixed"]["stop"]) - GUARD
        index = np.arange(start, stop)
        if stop <= start:
            raise ValueError("Guard consumed the supported interval")
        absolute = data["mcs6_resource_start"][word] + index
        slots, bins = np.divmod(absolute, len(NONPILOT))
        target = pn[index]
        cases.append((word, row, observed[index], target, slots + 2, bins))
    return cases


def phase_folds(observed, known, slots):
    """Choose a quarter turn on one slot parity and evaluate the other."""
    folds = []
    for fit_parity in (0, 1):
        fit = slots % 2 == fit_parity
        validate = ~fit
        if not np.any(fit) or not np.any(validate):
            folds.append({"fit_slot_parity": fit_parity, "status": "insufficient_slots"})
            continue
        signs = 1 - 2 * known[fit].astype(float)
        fit_score = (np.sum(observed[fit] * signs) * QUARTERS).real
        maxima = np.flatnonzero(np.isclose(fit_score, fit_score.max(), rtol=1e-6, atol=1e-6))
        if len(maxima) != 1:
            folds.append({"fit_slot_parity": fit_parity, "status": "ambiguous", "scores": fit_score.tolist()})
            continue
        turn = int(maxima[0])
        candidate = observed[validate, None] * QUARTERS
        margin = candidate.real
        threshold = 0.5 * np.median(abs(observed[fit]))
        ambiguous = abs(margin) <= threshold
        differences = ((margin < 0) != known[validate, None]) & ~ambiguous
        folds.append({"fit_slot_parity": fit_parity, "status": "tested", "turn": turn,
                      "scores": fit_score.tolist(), "train_slots": slots[fit].tolist(),
                      "validation_slots": slots[validate].tolist(),
                      "initial_hard_differences": int(np.count_nonzero((observed[validate].real < 0) != known[validate])),
                      "differences_by_turn": differences.sum(axis=0).tolist(),
                      "erasures_by_turn": ambiguous.sum(axis=0).tolist()})
    return folds


def audit_carriers(cases):
    counts = np.zeros((2, len(NONPILOT)), dtype=int)
    errors = np.zeros_like(counts)
    for _, row, observed, known, _, bins in cases:
        partition = int(not row["calibration"])
        different = (observed.real < 0) != known
        counts[partition] += np.bincount(bins, minlength=len(NONPILOT))
        errors[partition] += np.bincount(bins[different], minlength=len(NONPILOT))
    peak = np.flatnonzero(errors[0] == errors[0].max())
    if len(peak) != 1:
        raise ValueError("Calibration did not identify a unique maximum error carrier")
    peak = peak[0]
    rows = []
    for word, row, observed, known, slots, bins in cases:
        candidate = bins == peak
        rows.append({"word": word, "frame": row["frame"], "group": row["group"],
                     "calibration": row["calibration"], "target": known[candidate].tolist(),
                     "observed": [[float(x.real), float(x.imag)] for x in observed[candidate]],
                     "slots": slots[candidate].tolist(),
                     "phase_folds": phase_folds(observed[candidate], known[candidate], slots[candidate])})
    return {"carrier": int(NONPILOT[peak]), "carrier_axis": NONPILOT.tolist(),
            "partition_order": ["calibration", "reserved"],
            "matches_compared": counts.tolist(), "hard_differences": errors.tolist(), "words": rows}


def main():
    require_run_context()
    inputs = {}
    audit = read(checked_artifact("v034-qin-low-mcs-audit", "E036", "qin_low_mcs_audit.json", inputs))
    source = checked_artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_symbols.npz", inputs)
    code5 = checked_artifact("v022-ut-replay", "E014", "ut_observations.npz", inputs)
    model = audit["pn_model"]
    seq = pn_bits()
    indices = np.arange(K)
    p = equality_predictions(indices, seq, model)
    controls = {f"phase_{d:+d}": equality_predictions(indices, seq, {**model, "phase": model["phase"] + d})
                for d in (-1, 1)}
    rows = audit["formats"]["6"]["words"]
    with np.load(source, allow_pickle=False) as data:
        hard = (data["mcs6_symbols"][:, :K].real < 0).astype(np.uint8)
        words = [{"word": i, "frame": row["frame"], "group": row["group"], "calibration": row["calibration"],
                  "last": row["last"], "pn_fill_supported": row["fixed"]["accepted"],
                  "tests": grammar_controls(bits, p, controls)}
                 for i, (row, bits) in enumerate(zip(rows, hard, strict=True))]
        carrier = audit_carriers(padding_observations(data, rows, p))
    with np.load(code5, allow_pickle=False) as data:
        length = 1024
        transfer = [[record_candidate(np.packbits(bits ^ p[:length] ^ polarity, bitorder="big").tobytes())
                     for bits in data["decoded_code5_words"][:, :length]]
                    for polarity in (0, 1)]
    result = {"schema": "starlink-qin-mcs6-structure/v1",
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "pn_model": model, "systematic_boundary_candidate": K, "groups": audit["formats"]["6"]["groups"],
              "group_seed": audit["whole_group_seed"], "grammar_prefix_hex": PREFIX.hex(),
              "record_width_by_tag": RECORD_WIDTH, "min_c_records": MIN_C_RECORDS, "scan_limit_bytes": SCAN_LIMIT,
              "words": words, "code5_transfer_polarities": transfer,
              "pn_padding_guard_bits": GUARD, "carrier_audit": carrier}
    output = experiment_output("E037")
    output.mkdir(parents=True, exist_ok=True)
    (output / "qin_mcs6_structure.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for fitted in (True, False):
        group = [word for word in words if word["calibration"] == fitted]
        print("calibration" if fitted else "reserved", sum(w["tests"]["fixed"]["match"] for w in group),
              "grammar hits from", len(group), "words", flush=True)
    print("Selected edge carrier", carrier["carrier"], flush=True)


if __name__ == "__main__":
    main()
