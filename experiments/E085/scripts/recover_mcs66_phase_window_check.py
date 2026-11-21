"""Validate an additional independent MCS 66 check found in local phase windows."""

from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_mcs66_expanded_erasures import compare_proposals
from audit_mcs66_single_erasures import evaluate, repeat_partners
from audit_qam_pn15 import collect, equality_predictions
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from extend_mcs66_parity_matrix import padding_constant, raw_assessment
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, LIFT, fit_taps, held_only, packed_arrays, project_checks, totals
from search_mcs66_early_projections import PAIR_BITS, PAIR_ID_BITS, insert_support, matching_pairs, reduce_support, signatures_for
from sparse_code5_core import binary_rank


SEED = 2026112187
WIDTHS = (16, 32, 64)
DIFFERENCES = 24


def nominations(bits, mask, known):
    coordinates, signatures, _ = signatures_for(bits, mask, 1)
    blocks, phase = np.divmod(coordinates, LIFT)
    anchors = np.flatnonzero((blocks < 4) & (phase == 0))
    upper = [np.flatnonzero(blocks == b) for b in range(4, 8)]
    tables, hits = {}, set()
    for i, j, k in combinations(range(4), 3):
        if (i, j) not in tables:
            u, v = np.meshgrid(upper[i], upper[j], indexing="ij")
            values = signatures[u] ^ signatures[v]
            pairs = ((values << PAIR_BITS) | (u.astype(np.uint64) << PAIR_ID_BITS) | v.astype(np.uint64)).ravel()
            pairs.sort()
            tables[i, j] = pairs, pairs >> PAIR_BITS
        pairs, keys = tables[i, j]
        for anchor in anchors:
            targets = signatures[anchor] ^ signatures[upper[k]]
            for t, left, right in matching_pairs(pairs, keys, targets, np.uint64(0)):
                support = tuple(int(x) for x in coordinates[[anchor, left, right, upper[k][t]]])
                if reduce_support(support, known):
                    hits.add(support)
    return hits


def search(bits, mask, short, known):
    rng = np.random.default_rng(SEED)
    frequency, trials = Counter(), []
    coordinates = np.arange(bits.shape[1])
    for width in WIDTHS:
        for start in range(0, LIFT, width):
            pick = rng.permutation(short)
            other = np.resize(pick[1:], DIFFERENCES)
            rotation = rng.integers(start, start + width, size=DIFFERENCES)
            rotate = coordinates // LIFT * LIFT + (coordinates + rotation[:, None]) % LIFT
            right = other[:, None]
            values = bits[pick[0], rotate] ^ bits[right, rotate]
            seen = mask[pick[0], rotate] & mask[right, rotate]
            hits = nominations(values, seen, known)
            frequency.update(sorted(hits))
            trials.append({"start": start, "width": width, "reference_row": int(pick[0]),
                           "other_rows": other.tolist(), "rotations": rotation.tolist(), "nominations": len(hits)})
    return frequency, trials


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    meta = read(artifact(EXTRACTION, "E083", "mcs66_padding_words.json"))
    edge = meta["systematic_end"]
    with np.load(artifact(EXTRACTION, "E083", "mcs66_padding_words.npz"), allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ("bits", "mask"))
    with np.load(artifact("v082-mcs66-expanded-checks", "E084", "mcs66_expanded_checks.npz"), allow_pickle=False) as z:
        existing = z["row_support"]
    groups = np.array([r["group"] for r in meta["rows"]])
    fit = np.array([r["calibration"] for r in meta["rows"]])
    bound = np.array([r["bound"] for r in meta["rows"]])
    short = np.flatnonzero(fit & (bound <= LIFT) & (mask[:, edge:].mean(axis=1) >= .98))
    if len(np.unique(groups[short])) != len(short):
        raise ValueError("Short-word search must use one word per group")
    basis = {}
    for support in existing:
        insert_support(support[support >= edge] - edge, basis)
    prior_rank = len(basis)
    frequencies, trials = search(bits[:, edge:], mask[:, edge:], short, basis)
    repeated = [support for support, count in frequencies.items() if count >= 2]
    if len(repeated) != 1:
        raise ValueError(f"Expected one recurring shape, found {len(repeated)}")
    parity_shape = [divmod(x, LIFT) for x in repeated[0]]
    par = orbit_positions(parity_shape, LIFT)
    values, seen = project_checks(bits, mask, edge + par)
    pn = equality_predictions(np.arange(edge), pn_bits(), meta["pn_model"]).astype(bool)
    active = (bits[:, :edge] ^ pn) & mask[:, :edge] & (np.arange(edge) < bound[:, None])
    ref, trusted = padding_constant(values, seen, active, mask[:, :edge] | (np.arange(edge) >= bound[:, None]),
                                    [], groups, fit & (bound <= LIFT))
    systemic, trace = fit_taps(values ^ ref, seen & trusted, active.reshape(len(bits), -1, LIFT),
                               bound, groups, fit)
    sys_pos = orbit_positions(systemic, LIFT)
    checks = np.column_stack((sys_pos, edge + par))
    constants = np.bitwise_xor.reduce(pn[sys_pos], axis=-1)
    if trusted.sum() < 256 or np.any((constants ^ ref) & trusted):
        raise ValueError("Systematic taps fail the calibrated padding signs")

    packed = "v080-gain-padded-words"
    cohorts = read(artifact(packed, "E082", "gain_padded_words.json"))["cohorts"]
    with np.load(artifact(packed, "E082", "gain_padded_words.npz"), allow_pickle=False) as z:
        non66 = packed_arrays(z, ("nonterminal_66_bits", "nonterminal_66_seen"))
        non70 = held_only(*packed_arrays(z, ("nonterminal_70_bits", "nonterminal_70_seen")), cohorts["nonterminal_70"])
    train = np.array([r["calibration"] for r in cohorts["nonterminal_66"]])
    syndrome, obs = project_checks(non66[0][train], non66[1][train], checks)
    calibration = totals(syndrome, obs, np.array([r["group"] for r in cohorts["nonterminal_66"]])[train])
    if len(calibration["groups"]) < 8 or calibration["group_accuracy"] < .75:
        raise ValueError("Nonterminal calibration does not support the candidate")
    for positions in par:
        insert_support(positions, basis)
    support = np.vstack((existing, checks))
    matrix = np.zeros((len(support), bits.shape[1]), dtype=np.uint8)
    matrix[np.arange(len(support))[:, None], support] = 1
    rank = binary_rank(matrix)
    if rank != len(support) or len(basis) != rank:
        raise ValueError("The expanded matrix or its parity projection is rank deficient")

    raw_run = "v026-ut-qam-extract-repeat"
    raw, _, _ = collect(artifact(raw_run, "E028", "ut_qam_symbols.npz"),
                        read(artifact(raw_run, "E028", "ut_qam_extraction.json")))
    results = {}
    raw_assessment(results, raw, (sys_pos, edge + par), {
        "archive_terminal_held": (bits[~fit], mask[~fit], groups[~fit]),
        "archive_nonterminal_held": held_only(*non66, cohorts["nonterminal_66"]),
        "wrong_format_archive": non70,
    }, [int(v) if k else None for v, k in zip(ref, trusted, strict=True)])
    baseline = read(artifact("v082-mcs66-expanded-erasures", "E084", "mcs66_expanded_erasures.json"))
    repeats = read(artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    proposed, proposal_stats = evaluate(meta["rows"], bits, mask, support, pn, repeat_partners(repeats, meta["rows"]))
    held_before = [row for row in baseline["rows"] if not fit[row["row"]]]
    held_after = [row for row in proposed if not fit[row["row"]]]
    report = {
        "schema": "starlink-mcs66-phase-window-check/v1",
        "input_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()},
        "search_seed": SEED, "short_word_indices": short.tolist(), "short_groups": groups[short].tolist(),
        "search_trials": trials, "recurrence_histogram": dict(sorted(Counter(frequencies.values()).items())),
        "candidate_recurrences": frequencies[repeated[0]], "parity_shape": parity_shape,
        "systematic_shape": systemic, "onset_fit_trace": trace, "matching_pn_constants": int(trusted.sum()),
        "nonterminal_calibration": calibration, "evaluations": results,
        "prior_parity_rank": prior_rank, "parity_rank": len(basis), "full_rank": rank,
        "single_erasure_summary": proposal_stats,
        "held_single_erasure_change": compare_proposals(held_before, held_after, meta["rows"], pn),
        "single_erasure_proposals": proposed,
    }
    out = experiment_output("E085")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs66_phase_window_check.npz", row_support=support,
                        check_matrix_packed=np.packbits(matrix, axis=1))
    (out / "mcs66_phase_window_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(trials)} windows, candidate seen in {frequencies[repeated[0]]}, parity rank {prior_rank}->{rank}")


if __name__ == "__main__":
    main()
