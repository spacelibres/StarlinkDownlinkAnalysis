"""Record the bounded parity-only search for a shared extra parity-block relation."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_mcs_lower_rate_core import BIT_LENGTH, TARGETS
from audit_ut_repeated_data import checked_artifact, read
from recover_mcs66_systematic_checks import LIFT, packed_arrays
from search_mcs66_early_projections import PAIR_BITS, PAIR_ID_BITS, matching_pairs, signatures_for


ANCHOR = 11776
SEED = 20261121132
WIDTHS = (16, 32, 64)
DIFFERENCES = 24


def anchored_matches(differences, seen):
    coords, signatures, _ = signatures_for(differences, seen, 2)
    anchor = np.flatnonzero(coords == 0)
    if not len(anchor):
        return set()
    value = signatures[anchor[0]]
    blocks = [np.flatnonzero(coords // LIFT == j) for j in range(9)]
    matches = set()
    for j, k in combinations(range(1, 9), 2):
        a, b = blocks[j], blocks[k]
        ordered = a[np.argsort(signatures[a])]
        query = value ^ signatures[b]
        lo = np.searchsorted(signatures[ordered], query, side="left")
        hi = np.searchsorted(signatures[ordered], query, side="right")
        for index in np.flatnonzero(hi > lo):
            matches.update((0, int(coords[u]), int(coords[b[index]])) for u in ordered[lo[index]:hi[index]])
        if k == 8:
            continue
        u, v = np.meshgrid(a, b, indexing="ij")
        pairs = (((signatures[u] ^ signatures[v]) << PAIR_BITS)
                 | (u.astype(np.uint64) << PAIR_ID_BITS) | v.astype(np.uint64)).ravel()
        pairs.sort()
        keys = pairs >> PAIR_BITS
        for last in blocks[k + 1:]:
            hits = matching_pairs(pairs, keys, value ^ signatures[last], np.uint64(0))
            matches.update((0, int(coords[i]), int(coords[h]), int(coords[last[n]])) for n, i, h in hits)
    return matches


def main():
    require_run_context()
    inputs = {}
    run = "v080-gain-padded-words"
    metadata = read(checked_artifact(run, "E082", "gain_padded_words.json", inputs))["cohorts"]
    groups = {}
    with np.load(checked_artifact(run, "E082", "gain_padded_words.npz", inputs), allow_pickle=False) as z:
        for code in TARGETS:
            name = f"short_{code}"
            rows = metadata[name]
            b, m = packed_arrays(z, (name + "_bits", name + "_seen"))
            b, m = b[:, ANCHOR:BIT_LENGTH], m[:, ANCHOR:BIT_LENGTH]
            eligible = np.flatnonzero(np.array([r["calibration"] and r["group"] != 19 for r in rows]) & (m.mean(1) > .98))
            pairs = [(int(i), int(j)) for i, j in combinations(eligible, 2) if rows[i]["group"] != rows[j]["group"]]
            groups[code] = b, m, pairs
    rng = np.random.default_rng(SEED)
    coords = np.arange(BIT_LENGTH - ANCHOR)
    counter, windows = Counter(), []
    for width in WIDTHS:
        for start in range(0, LIFT, width):
            samples, seen, traces = [], [], []
            for _ in range(DIFFERENCES):
                code = TARGETS[rng.integers(len(TARGETS))]
                b, m, pairs = groups[code]
                i, j = pairs[rng.integers(len(pairs))]
                r = int(rng.integers(start, start + width))
                positions = coords // LIFT * LIFT + (coords + r) % LIFT
                samples.append(b[i, positions] ^ b[j, positions])
                seen.append(m[i, positions] & m[j, positions])
                traces.append([code, i, j, r])
            hits = anchored_matches(np.stack(samples), np.stack(seen))
            counter.update(sorted(hits))
            windows.append({"rotation_interval": [start, start + width], "difference_choices": traces, "nominations": len(hits)})
    histogram = Counter(counter.values())
    result = {"schema": "starlink-lower-rate-parity-only-probe/v1", "seed": SEED,
              "anchor_absolute_coordinate": ANCHOR, "windows": windows, "recurrence_histogram": dict(sorted(histogram.items())),
              "repeated_candidates": [{"absolute_positions": [ANCHOR + q for q in p], "votes": n}
                                       for p, n in sorted(counter.items()) if n > 1],
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    out = experiment_output("E132")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lower_rate_parity_only_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(len(windows), "rotation windows, recurrence count histogram", dict(sorted(histogram.items())), flush=True)


if __name__ == "__main__":
    main()
