"""Apply the E130 all-origins visibility test to the recovered MCS 70/88 code."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_leo_full_code_observability import GROUP_SIZE, disjoint_bases, observed_bits
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from recover_mcs70_full_code import BIT_LENGTH, SYSTEMATIC


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    geometry = read(artifact("v128-leo-full-code-observability", "E130", "leo_full_code_observability.json"))
    mask = np.isin(NONPILOT, geometry["union_carriers"])
    with np.load(artifact("v129-mcs70-full-code", "E131", "mcs70_full_checks.npz"), allow_pickle=False) as z:
        checks = np.unpackbits(z["projected_matrix"], axis=1)
    rank = BIT_LENGTH - SYSTEMATIC
    columns = [int.from_bytes(row.tobytes(), "little") for row in np.packbits(checks.T, axis=1, bitorder="little")]
    packed, summary = {}, {}
    for code, width in ((70, 4), (88, 5)):
        visibility = observed_bits(mask, np.arange(len(mask)), width)
        certificates = [disjoint_bases(columns, ~visibility[i:i + GROUP_SIZE].any(axis=0), rank)
                        for i in range(0, len(mask), GROUP_SIZE)]
        if any(len(left) != rank or len(right) != rank for left, right in certificates):
            raise ValueError(f"Could not certify an erased basis for every MCS {code} origin")
        packed[f"mcs{code}_basis_columns"] = np.asarray(certificates, dtype=np.uint16)
        counts = visibility.sum(axis=1)
        summary[code] = {"origins": len(mask), "groups": len(certificates), "certified_erasure_rank": rank,
                         "compatible_word_dimension_range": [int(SYSTEMATIC - counts.max()),
                                                             int(SYSTEMATIC - counts.min())]}
    result = {"schema": "starlink-mcs70-leo-observability/v1", "origin_group_size": GROUP_SIZE,
              "union_carriers": geometry["union_carriers"], "results": summary,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    out = experiment_output("E131")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "leo_mcs70_certificates.npz", **packed)
    (out / "leo_mcs70_observability.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
