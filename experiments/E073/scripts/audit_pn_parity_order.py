"""Test whether measured PN15 systematics are needed to reproduce received parity."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_legacy_pn_transfer import LEGACY, MIN_LENGTH, strongest
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


def byte_run(differences):
    """Return the longest exact run of complete bytes with its end exclusive."""
    complete = differences[: len(differences) // 8 * 8]
    exact = ~np.any(complete.reshape(-1, 8), axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, exact, False]))
    if not len(edges):
        return 0, 0
    begins, ends = edges[::2], edges[1::2]
    longest = np.argmax(ends - begins)
    return int(8 * begins[longest]), int(8 * ends[longest])


def parity_intervention(system, received, matrix, interval, replacements):
    baseline = (system @ matrix) & 1
    result = {"observed": {"mismatches": int(np.count_nonzero(baseline ^ received)), "changed_predictions": 0}}
    for name, replacement in replacements.items():
        delta = ((system[interval] ^ replacement) @ matrix[interval]) & 1
        result[name] = {
            "mismatches": int(np.count_nonzero(baseline ^ delta ^ received)),
            "changed_predictions": int(delta.sum()),
        }
    return result


def audit_words(bits, records, matrix, parity_start, model, sequence, prior_scope):
    width, checks = matrix.shape
    predicted = equality_predictions(np.arange(parity_start + checks), sequence, model)
    shifted = {
        f"phase_{shift:+d}": equality_predictions(np.arange(width), sequence, {**model, "phase": model["phase"] + shift})
        for shift in (-1, 1)
    }
    full_mask_delta = None
    if parity_start == width:
        full_mask_delta = ((predicted[:width] @ matrix) & 1) ^ predicted[width:]
    rows = []
    for i, (word, record) in enumerate(zip(bits, records, strict=True)):
        systematic = word[:width]
        received = word[parity_start:parity_start + checks]
        differences = systematic ^ predicted[:width]
        support = strongest(differences)
        lo, hi = byte_run(differences)
        scope = prior_scope(record)
        row = {"index": i, "frame": record.get("frame", record.get("candidate_index")), "word": record["word"],
               "start_resource": record["start_resource"], "prior_scope": scope,
               "systematic_pn_max_run": support, "exact_byte_interval": [lo, hi],
               "retained_for_intervention": bool(support["accepted_144"] and hi - lo >= MIN_LENGTH)}
        rows.append(row)
        interval = slice(lo, hi)
        alternatives = {}
        if row["retained_for_intervention"]:
            alternatives = {"zero": np.zeros(hi - lo, dtype=np.uint8),
                            "complement": systematic[interval] ^ 1,
                            **{name: value[interval] for name, value in shifted.items()}}
            row["interval_ends_at_systematic_boundary"] = hi == width
        comparisons = parity_intervention(systematic, received, matrix, interval, alternatives)
        if full_mask_delta is not None and row["retained_for_intervention"]:
            syndrome = ((systematic @ matrix) & 1) ^ received
            comparisons["full_word_pn_xor"] = {
                "mismatches": int(np.count_nonzero(syndrome ^ full_mask_delta)),
                "changed_predictions": int(full_mask_delta.sum()),
            }
        row["predictions"] = comparisons
    counts = Counter()
    for row in rows:
        counts["examined"] += 1
        if row["retained_for_intervention"]:
            counts["pn_interval"] += 1
            if not row["predictions"]["observed"]["mismatches"]:
                counts["pn_interval_with_exact_raw_parity"] += 1
                counts["exact_at_boundary" if row["interval_ends_at_systematic_boundary"] else "exact_interior"] += 1
                if row["predictions"]["zero"]["mismatches"]:
                    counts["exact_raw_parity_disrupted_by_zero"] += 1
    return {"systematic_width": width, "parity_start": parity_start, "parity_width": checks,
            "full_word_pn_xor_syndrome_weight": None if full_mask_delta is None else int(full_mask_delta.sum()),
            "counts": dict(counts), "words": rows}


def main():
    require_run_context()
    inputs = {}
    previous = read(checked_artifact("v030-legacy-pn-transfer-final", "E032", "legacy_pn_transfer.json", inputs))
    model = previous["pn_model"]
    sequence = pn_bits()
    results = {}
    matrices = {}
    for code in (5, 9, 21, 37):
        run, experiment, stem, key, sidecar, width = LEGACY[code]
        report = read(checked_artifact(run, experiment, f"{sidecar}.json", inputs))
        with np.load(checked_artifact(run, experiment, f"{stem}.npz", inputs), allow_pickle=False) as z:
            bits = z[key]
            matrices[code] = z["parity_generator"] if code != 37 else matrices[21][384:]
        if code == 5:
            scope = lambda row: "matrix_fit" if row["training"] else "matrix_held"
        elif code == 9:
            fit = set(report["published_groups"]["training_frames"])
            scope = lambda row: "published_matrix_fit" if row["frame"] in fit else "published_matrix_held"
        elif code == 21:
            fit = set(report["published_group"]["fit_frames"])
            scope = lambda row: "published_matrix_fit" if row["frame"] in fit else "published_matrix_held"
        else:
            fit = set(report["training_frames"])
            scope = lambda row: "shortened_label_fit" if row["frame"] in fit else "shortened_label_held"
        results[f"legacy_{code}"] = audit_words(bits, report["words"], matrices[code],
                                                 5120 if code == 37 else width, model, sequence, scope)
    capture_run = "v022-ut-replay"
    report = read(checked_artifact(capture_run, "E014", "ut_transfer.json", inputs))
    with np.load(checked_artifact(capture_run, "E014", "ut_observations.npz", inputs), allow_pickle=False) as z:
        results["ut_5"] = audit_words(z["hard_code5_words"], report["words"], matrices[5], 1024, model,
                                      sequence, lambda _: "external_capture_for_code5_matrix")
    data = {"schema": "starlink-pn-parity-order/v1",
            "inputs_sha256": {path.relative_to(ROOT).as_posix(): value for path, value in inputs.items()},
            "pn_model": model, "minimum_exact_bytes": MIN_LENGTH // 8, "sets": results}
    output = experiment_output("E073")
    output.mkdir(parents=True, exist_ok=True)
    (output / "pn_parity_order.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for key, result in results.items():
        print(key, result["counts"], "global PN xor check weight", result["full_word_pn_xor_syndrome_weight"])


if __name__ == "__main__":
    main()
