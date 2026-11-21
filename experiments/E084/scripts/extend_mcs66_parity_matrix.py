"""Complete new projected checks and admit only rank-increasing homogeneous rows."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import collect, equality_predictions
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import (EXTRACTION, LIFT, fit_taps, held_only, packed_arrays, project_checks,
                                             score_word_set, totals)
from search_mcs66_early_projections import PREFIX, insert_support, reduce_support
from sparse_code5_core import binary_rank


PROJECTIONS = "v082-mcs66-early-projections"


def exact_completions(dictionary, target):
    if not target.any():
        return [()]
    patterns = [int.from_bytes(row.tobytes(), "little") for row in np.packbits(dictionary, axis=1)]
    goal = int.from_bytes(np.packbits(target).tobytes(), "little")
    locations = defaultdict(list)
    for i, vector in enumerate(patterns):
        locations[vector].append(i)
    one = [(i,) for i in locations.get(goal, [])]
    two = [(i, j) for i, v in enumerate(patterns) for j in locations.get(v ^ goal, []) if j > i]
    return one + two


def empty_check_set(bits):
    empty = np.zeros((len(bits), LIFT), dtype=bool)
    return empty, ~empty


def padding_constant(parity, parity_seen, active, active_seen, early, groups, train):
    if early:
        terms, known = project_checks(active, active_seen, orbit_positions(early, LIFT))
    else:
        terms, known = empty_check_set(active)
    voted = parity ^ terms
    visible = parity_seen & known
    votes = np.zeros(LIFT)
    count = np.zeros(LIFT, dtype=int)
    for group in np.unique(groups[train]):
        use = train & (groups == group)
        n = visible[use].sum(axis=0)
        k = (visible[use] & voted[use]).sum(axis=0)
        votes += (2 * k - n) / np.maximum(n, 1)
        count += n > 0
    return votes > 0, (count >= 8) & (abs(votes) / np.maximum(count, 1) >= .8)


def complete_shape(seeds, bits, mask, pn, bound, groups, fit, ordinary):
    edge = len(pn)
    active = (bits[:, :edge] ^ pn) & mask[:, :edge] & (np.arange(edge) < bound[:, None])
    active_seen = mask[:, :edge] | (np.arange(edge) >= bound[:, None])
    pn_dictionary = pn[orbit_positions([divmod(i, LIFT) for i in range(edge)], LIFT)].T
    results = []
    for seed in seeds:
        parity_shape = seed["parity_shape"]
        parity = edge + orbit_positions(parity_shape, LIFT)
        par_value, visible = project_checks(bits, mask, parity)
        ref, ref_seen = padding_constant(par_value, visible, active, active_seen, seed["early_shape"],
                                         groups, fit & (bound <= PREFIX))
        receipt = {"parity_shape": parity_shape, "early_shape": seed["early_shape"],
                   "matching_rotations": seed["matching_rotations"], "calibrated_signs": int(ref_seen.sum()),
                   "completions": []}
        results.append(receipt)
        if ref_seen.sum() < 256:
            continue
        recovered, _ = fit_taps(par_value ^ ref, visible & ref_seen, active.reshape(len(bits), -1, LIFT),
                                bound, groups, fit)
        receipt["onset_systematic_shape"] = recovered
        pn_value = np.bitwise_xor.reduce(pn_dictionary[[a * LIFT + b for a, b in recovered]], axis=0)
        goal = (pn_value ^ ref)[ref_seen]
        receipt["onset_only_constant_agreements"] = int((~goal).sum())
        for additions in exact_completions(pn_dictionary[:, ref_seen], goal):
            supplement = [divmod(i, LIFT) for i in additions]
            shape = sorted(set(recovered) ^ set(supplement))
            if not shape:
                continue
            sys = orbit_positions(shape, LIFT)
            check = np.column_stack((sys, parity))
            syndrome, obs = project_checks(ordinary[0], ordinary[1], check)
            score = totals(syndrome, obs, ordinary[2])
            receipt["completions"].append({"supplement": supplement, "systematic_shape": shape, "fit_nonterminal": score})
        receipt["fixed_affine_constants"] = [int(value) if enough else None
                                              for value, enough in zip(ref, ref_seen, strict=True)]
    return results


def raw_assessment(report, raw, samples, tests, reference):
    ref = np.array([bool(x) for x in reference])
    seen = np.array([x is not None for x in reference])
    for name, observations in tests.items():
        report[name] = score_word_set(*observations, *samples, ref, seen)
    for mcs, name in ((66, "raw_250"), (70, "wrong_format_raw")):
        refined, baseline, _, rows = raw[mcs]
        groups = np.array([r["frame"] for r in rows])
        for front_end, bits in (("baseline", baseline), ("refined", refined)):
            report[name + "_" + front_end] = score_word_set(bits.astype(bool), np.ones(bits.shape, bool),
                                                            groups, *samples, ref, seen)


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    terminal = load(EXTRACTION, "E083", "mcs66_padding_words.json")
    edge, n = terminal["systematic_end"], terminal["codeword_bits"]
    source = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    with np.load(source, allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ("bits", "mask"))
    pn = equality_predictions(np.arange(edge), pn_bits(), terminal["pn_model"]).astype(bool)
    groups = np.array([r["group"] for r in terminal["rows"]])
    fit = np.array([r["calibration"] for r in terminal["rows"]])
    bound = np.array([r["bound"] for r in terminal["rows"]])
    cohorts = load("v080-gain-padded-words", "E082", "gain_padded_words.json")["cohorts"]
    packed = checked_artifact("v080-gain-padded-words", "E082", "gain_padded_words.npz", inputs)
    with np.load(packed, allow_pickle=False) as z:
        non66 = packed_arrays(z, ("nonterminal_66_bits", "nonterminal_66_seen"))
        non70 = held_only(*packed_arrays(z, ("nonterminal_70_bits", "nonterminal_70_seen")), cohorts["nonterminal_70"])
    non66_rows = cohorts["nonterminal_66"]
    nfit = np.array([r["calibration"] for r in non66_rows])
    fit_ordinary = non66[0][nfit], non66[1][nfit], np.array([r["group"] for r in non66_rows])[nfit]
    projections = load(PROJECTIONS, "E084", "mcs66_early_projections.json")
    attempts = complete_shape(projections["retained"], bits, mask, pn, bound, groups, fit, fit_ordinary)
    good = [(row, model) for row in attempts for model in row["completions"]
            if len(model["fit_nonterminal"]["groups"]) >= 8 and model["fit_nonterminal"]["group_accuracy"] >= .75]
    good.sort(key=lambda k: (-k[1]["fit_nonterminal"]["group_accuracy"], -k[0]["matching_rotations"], k[0]["parity_shape"]))
    prior_path = checked_artifact("v081-mcs66-systematic-checks", "E083", "mcs66_partial_checks.npz", inputs)
    with np.load(prior_path, allow_pickle=False) as z:
        existing = z["row_support"]
    basis = {}
    for old in existing:
        insert_support(old[old >= edge] - edge, basis)
    initial = len(basis)
    matrices = [existing]
    selected = []
    for row, model in good:
        parity = orbit_positions(row["parity_shape"], LIFT)
        if not reduce_support(parity[0], basis):
            continue
        old_rank = len(basis)
        for s in parity:
            insert_support(s, basis)
        sys = orbit_positions(model["systematic_shape"], LIFT)
        matrices.append(np.column_stack((sys, edge + parity)))
        selected.append((row, model, sys, edge + parity, len(basis) - old_rank))
    # Held and raw waveform samples only enter after the complete list is frozen.
    raw_run = "v026-ut-qam-extract-repeat"
    raw_path = checked_artifact(raw_run, "E028", "ut_qam_symbols.npz", inputs)
    raw, _, _ = collect(raw_path, load(raw_run, "E028", "ut_qam_extraction.json"))
    tests = {"archive_terminal_held": (bits[~fit], mask[~fit], groups[~fit]),
             "archive_nonterminal_held": held_only(*non66, non66_rows), "wrong_format_archive": non70}
    verified = []
    for row, model, sys, parity, gained in selected:
        receipt = {"parity_shape": row["parity_shape"], "early_shape": row["early_shape"],
                   **model, "calibrated_signs": row["calibrated_signs"], "additional_parity_rank": gained}
        raw_assessment(receipt, raw, (sys, parity), tests, row["fixed_affine_constants"])
        verified.append(receipt)
        print("selected", receipt["parity_shape"], "added rank", gained, "systematic", model["systematic_shape"], flush=True)
    full = np.vstack(matrices)
    matrix = np.zeros((len(full), n), dtype=np.uint8)
    matrix[np.arange(len(full))[:, None], full] = 1
    rank = binary_rank(matrix)
    result = {"schema": "starlink-mcs66-expanded-matrix/v1", "prior_parity_rank": initial, "full_rank": rank,
              "parity_rank": len(basis), "codeword_bits": n, "candidate_systematic_end": edge,
              "calibration_attempts": attempts, "new_check_families": verified,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    out = experiment_output("E084")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs66_expanded_checks.npz", row_support=full, check_matrix_packed=np.packbits(matrix, axis=1))
    (out / "mcs66_expanded_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Final GF(2) rank", rank, "parity rank", len(basis))


if __name__ == "__main__":
    main()
