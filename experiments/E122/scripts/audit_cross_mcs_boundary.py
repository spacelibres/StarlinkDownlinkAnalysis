"""Test serial origin on QC boundary rows and transfer all five MCS 66 families."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import LIFT, checks, paired_syndromes, score_cohort
from audit_ut_repeated_data import checked_artifact, read


def score_boundaries(bits, mask, groups, support):
    rows = []
    for family in support:
        indices = np.flatnonzero(np.any(family % LIFT == LIFT - 1, axis=1))
        boundary = family[indices]
        direct, shifted = paired_syndromes(bits, mask, groups, boundary, boundary + 1)
        rows.append({"boundary_rows": indices.tolist(), "direct": direct, "shifted": shifted})
    return rows


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    matrices = checks(artifact)
    with np.load(artifact("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz"),
                 allow_pickle=False) as data:
        full66 = np.sort(data["row_support"], axis=1).reshape(5, LIFT, 14)
    if not np.array_equal(full66[:2], matrices[66]):
        raise ValueError("E085 matrix prefix differs from E083")
    matrices[66] = full66
    run = "v120-cross-mcs-checks"
    result = read(artifact(run, "E122", "cross_mcs_checks.json"))
    source = artifact(run, "E122", "cross_mcs_observations.npz")
    histogram = {}
    with np.load(source, allow_pickle=False) as archive:
        for origin in ("archive", "raw250"):
            for label in (71, 75, 78, 81, 85, 88, 92, 93):
                for kind in ("nonterminal", "terminal"):
                    name = f"{label}_{kind}"
                    cohort = result[origin].get(name)
                    if not cohort or "reserved" not in cohort:
                        continue
                    rows = cohort["rows"]
                    held = np.array([not r["calibration"] for r in rows])
                    bits, mask = (
                        np.unpackbits(archive[f"{origin}_{name}_{k}"][held], axis=1)
                        for k in ("bits", "seen")
                    )
                    mask = mask.astype(bool)
                    selected = [row for row, keep in zip(rows, held, strict=True) if keep]
                    groups = np.array([row["group"] for row in selected])
                    key = f"{origin}_{name}"
                    histogram[key] = {
                        "words": len(selected), "groups": len(np.unique(groups)),
                        "checks": score_cohort(bits, mask, selected, matrices),
                        "boundary": {code: score_boundaries(bits, mask, groups, families)
                                     for code, families in matrices.items()},
                    }
    out = experiment_output("E122")
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "starlink-cross-mcs-boundary/v1", "cohorts": histogram,
              "artifact_input_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in hashes.items()}}
    (out / "cross_mcs_boundary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, matrix in ((85, 66), (88, 70)):
        row = histogram[f"archive_{label}_nonterminal"]
        print(f"MCS {label}, family zero-syndrome scores",
              [s["direct"]["group_accuracy"] for s in row["checks"][str(matrix)]])
        print("Block-boundary rows, original and shifted zero-syndrome scores",
              [(s["direct"]["group_accuracy"], s["shifted"]["group_accuracy"]) for s in row["boundary"][matrix]])


if __name__ == "__main__":
    main()
