"""Quantify the change in real erasure proposals from the expanded matrix."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_mcs66_single_erasures import evaluate, repeat_partners
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, packed_arrays


def compare_proposals(before, after, meta, pn):
    previous = {r["row"]: {p: v for p, v, _ in r["candidates"]} for r in before}
    revised = {r["row"]: {p: v for p, v, _ in r["candidates"]} for r in after}
    counts = Counter()
    for row in sorted(previous.keys() | revised.keys()):
        left = previous.get(row, {})
        right = revised.get(row, {})
        for pos in left.keys() & right.keys():
            if left[pos] != right[pos]:
                raise ValueError("Adding checks cannot reverse a unanimous supported bit")
            counts["retained"] += 1
        counts["withdrawn"] += len(left.keys() - right.keys())
        for pos in right.keys() - left.keys():
            counts["new"] += 1
            if meta[row]["bound"] <= pos < len(pn):
                counts["new_pn_compared"] += 1
                counts["new_pn_matches"] += int(right[pos] == pn[pos])
    return dict(sorted(counts.items()))


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    meta = load(EXTRACTION, "E083", "mcs66_padding_words.json")
    rows = meta["rows"]
    source = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    with np.load(source, allow_pickle=False) as z:
        bits, seen = packed_arrays(z, ("bits", "mask"))
    source = checked_artifact("v082-mcs66-expanded-checks", "E084", "mcs66_expanded_checks.npz", inputs)
    with np.load(source, allow_pickle=False) as z:
        support = z["row_support"]
    baseline = load("v081-mcs66-single-erasures", "E083", "mcs66_single_erasures.json")
    repeats = load("v079-code5-gain-masked", "E081", "code5_gain_transfer.json")
    links = repeat_partners(repeats, rows)
    pn = equality_predictions(np.arange(meta["systematic_end"]), pn_bits(), meta["pn_model"])
    predicted, summary = evaluate(rows, bits, seen, support, pn, links)
    changed = compare_proposals(baseline["rows"], predicted, rows, pn)
    report = {"schema": "starlink-mcs66-expanded-erasures/v1", "summary": summary, "change": changed,
              "baseline_summary": baseline["summary"], "rows": predicted,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E084")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mcs66_expanded_erasures.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                           encoding="utf-8")
    print(summary, changed)


if __name__ == "__main__":
    main()
