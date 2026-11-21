"""Check whether a search across rotated inter-word differences adds parity rank."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, LIFT, packed_arrays
from search_mcs66_early_projections import (PREFIX, insert_support, matching_pairs, normalized_shape, reduce_support,
                                           signatures_for, signed_pair_index)


SEED = 2026112185
TRIALS = 5


def paired_differences(values, seen, reference, comparisons, shifts, lift):
    columns = np.arange(values.shape[1])
    index = columns // lift * lift + (columns + shifts[:, None]) % lift
    return (values[reference, index] ^ values[comparisons[:, None], index],
            seen[reference, index] & seen[comparisons[:, None], index])


def projected_candidates(values, masks):
    coordinates, signs, all_ones = signatures_for(values, masks, 1)
    par = np.flatnonzero(coordinates >= PREFIX)
    anchors = par[(coordinates[par] - PREFIX) % LIFT == 0]
    table = defaultdict(list)
    for j, signature in enumerate(signs):
        table[int(signature)].append(j)
    candidates = set()

    def add(ix):
        if len(set(ix)) != len(ix) or np.bitwise_xor.reduce(signs[list(ix)]) != 0:
            return
        support = sorted(coordinates[list(ix)].tolist())
        if sum(k >= PREFIX for k in support) >= 2:
            candidates.add(normalized_shape(support))

    pairs, keys = signed_pair_index(signs, np.arange(len(signs)), all_ones)
    for i in anchors:
        targets = signs[i] ^ signs
        for j, k, h in matching_pairs(pairs, keys, targets, all_ones):
            add((i, j, k, h))
        for j, target in enumerate(targets):
            for k in table.get(int(target), []):
                add((i, j, k))
    return candidates, len(signs), len(anchors)


def audit(values, seen, rows, edge, support):
    upper = np.array([row["bound"] for row in rows])
    split = np.array([row["calibration"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    basis = {}
    for row in support:
        insert_support(row[row >= edge] - edge, basis)
    rng = np.random.default_rng(SEED)
    results = []
    for limit in (512, 1024):
        pool = np.flatnonzero(split & (upper <= limit) & (seen[:, PREFIX:].mean(axis=1) >= .98))
        pool = sorted(pool, key=lambda i: (rows[i]["pn_errors"] / max(rows[i]["pn_seen"], 1), i))
        chosen_groups = {}
        for index in pool:
            chosen_groups.setdefault(groups[index], index)
        unique = list(chosen_groups.values())
        nominations, traces = Counter(), []
        for _ in range(TRIALS):
            ref = int(rng.choice(unique[:6]))
            other = np.resize(rng.permutation([i for i in unique if i != ref]), 32)
            shifts = rng.integers(0, LIFT, size=len(other))
            value, available = paired_differences(values, seen, ref, other, shifts, LIFT)
            available[:, limit:PREFIX] = False
            choices, columns, anchors = projected_candidates(value, available)
            nominations.update(choices)
            traces.append({"reference_word": ref, "matched_words": other.tolist(), "rotations": shifts.tolist(),
                           "columns": columns, "anchors": anchors, "nominations": len(choices)})
        candidates = []
        for (par, early), count in sorted(nominations.items(), key=lambda r: (-r[1], r[0])):
            independent = bool(reduce_support(orbit_positions(par, LIFT)[0], basis))
            candidates.append({"parity_shape": par, "early_shape": early, "independent": independent,
                               "trials": count})
        results.append({"maximum_bound": limit, "groups": len(unique), "trials": traces,
                        "recurring_independent": sum(r["independent"] and r["trials"] >= 2 for r in candidates),
                        "nominations": candidates})
        print(limit, "groups", len(unique), "recurring independent", results[-1]["recurring_independent"], flush=True)
    return results


def main():
    require_run_context()
    inputs = {}
    info = read(checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.json", inputs))
    source = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    matrix = checked_artifact("v082-mcs66-expanded-checks", "E084", "mcs66_expanded_checks.npz", inputs)
    with np.load(source, allow_pickle=False) as z:
        bits, masks = packed_arrays(z, ("bits", "mask"))
    with np.load(matrix, allow_pickle=False) as z:
        support = z["row_support"]
    edge = info["systematic_end"]
    pn = equality_predictions(np.arange(PREFIX), pn_bits(), info["pn_model"]).astype(bool)
    before = np.arange(PREFIX) < np.array([r["bound"] for r in info["rows"]])[:, None]
    values = np.column_stack(((bits[:, :PREFIX] ^ pn) & before, bits[:, edge:]))
    seen = np.column_stack((masks[:, :PREFIX] | ~before, masks[:, edge:]))
    result = {"schema": "starlink-mcs66-rotation-differences/v1", "seed": SEED,
              "results": audit(values, seen, info["rows"], edge, support),
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E084")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mcs66_rotation_differences.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
