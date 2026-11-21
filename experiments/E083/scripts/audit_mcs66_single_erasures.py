"""Use the partial MCS 66 checks to nominate single-erasure replacements."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, packed_arrays


def single_erasure_estimates(bits, mask, support):
    missing = ~mask[support]
    usable = missing.sum(axis=1) == 1
    absent = missing[usable]
    locations = support[usable]
    remaining = np.where(absent, 0, bits[locations])
    votes = defaultdict(list)
    for coordinate, value in zip(locations[absent], np.bitwise_xor.reduce(remaining, axis=1), strict=True):
        votes[int(coordinate)].append(int(value))
    unanimous = {pos: values[0] for pos, values in votes.items() if min(values) == max(values)}
    count = {pos: len(votes[pos]) for pos in unanimous}
    return unanimous, count, sorted(votes.keys() - unanimous.keys())


def repeat_partners(repeats, rows):
    indices = {(row["frame"], row["resource"]): i for i, row in enumerate(rows)}
    links = defaultdict(set)
    for pair in repeats["pairs"]:
        if not pair["same_first_code5"]:
            continue
        for allocation in pair["allocations"]:
            if allocation["label"] != 66:
                continue
            for word in allocation["repeated_words"]:
                children = [(frame, word["resource"]) for frame in pair["frames"]]
                if all(child in indices for child in children):
                    left, right = map(indices.__getitem__, children)
                    links[left].add(right)
                    links[right].add(left)
    return links


def evaluate(rows, bits, mask, support, pn, links):
    counts = Counter()
    records = []
    for i, row in enumerate(rows):
        if row["calibration"]:
            continue
        estimates, votes, conflicts = single_erasure_estimates(bits[i], mask[i], support)
        counts.update({"proposed": len(estimates), "conflicting": len(conflicts),
                       "supported_twice": sum(value > 1 for value in votes.values())})
        evidence = []
        for p, v in sorted(estimates.items()):
            evidence.append([p, v, votes[p]])
            if row["bound"] <= p < len(pn):
                counts["pn_comparisons"] += 1
                counts["pn_matches"] += int(v == pn[p])
                counts["pn_constant_zero_matches"] += int(pn[p] == 0)
                if votes[p] > 1:
                    counts["pn_multiple_checks"] += 1
                    counts["pn_multiple_matches"] += int(v == pn[p])
            sources = {int(bits[j, p]) for j in links.get(i, ()) if mask[j, p]}
            if len(sources) == 1:
                counts["repeat_comparisons"] += 1
                counts["repeat_matches"] += int(v in sources)
            if len(sources) > 1:
                counts["repeat_conflicts"] += 1
        if estimates or conflicts:
            records.append({"row": i, "frame": row["frame"], "resource": row["resource"],
                            "candidates": evidence, "conflicting": conflicts})
    return records, dict(sorted(counts.items()))


def main():
    require_run_context()
    inputs = {}
    words = read(checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.json", inputs))
    source = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    with np.load(source, allow_pickle=False) as data:
        bits, mask = packed_arrays(data, ("bits", "mask"))
    check_source = checked_artifact("v081-mcs66-systematic-checks", "E083", "mcs66_partial_checks.npz", inputs)
    with np.load(check_source, allow_pickle=False) as data:
        support = data["row_support"]
    repeats = read(checked_artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json", inputs))
    links = repeat_partners(repeats, words["rows"])
    pn = equality_predictions(np.arange(words["systematic_end"]), pn_bits(), words["pn_model"])
    records, summary = evaluate(words["rows"], bits, mask, support, pn, links)
    report = {"schema": "starlink-mcs66-single-erasures/v1", "summary": summary, "rows": records,
              "verified_repeat_word_pairs": sum(map(len, links.values())) // 2,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E083")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mcs66_single_erasures.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
