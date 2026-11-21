"""Recover MCS 66 systematic taps using the PN onset and score raw syndromes."""

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
from sparse_code5_core import binary_rank


LIFT = 512
EXTRACTION = "v081-mcs66-padding-sweep"
MIN_EXPOSURE = 45.
MIN_ADVANTAGE = .24
MAX_TERMS_PER_BLOCK = 4


def cyclic_response(active, weights):
    left = np.fft.rfft(active, axis=-1)
    right = np.fft.rfft(weights, axis=-1)
    return np.fft.irfft(np.sum(left * np.conj(right), axis=0), n=active.shape[-1])


def project_checks(bits, mask, positions):
    return np.bitwise_xor.reduce(bits[:, positions], axis=-1), mask[:, positions].all(axis=-1)


def fit_taps(residual, admissible, activation, bound, groups, fit):
    shape, receipts = [], []
    residual = residual.copy()
    for block in range(activation.shape[1]):
        lower = LIFT * block
        upper = lower + LIFT
        use = fit & (bound > lower) & (bound <= upper)
        if len(np.unique(groups[use])) < 3:
            upper += LIFT
            use = fit & (bound > lower) & (bound <= upper)
        selected = {"block": block, "bound_interval": [lower, upper], "words": np.flatnonzero(use).tolist(),
                    "groups": int(len(np.unique(groups[use]))), "accepted": []}
        receipts.append(selected)
        if selected["groups"] < 2:
            continue
        _, inv, group_count = np.unique(groups[use], return_inverse=True, return_counts=True)
        group_weights = admissible[use] / group_count[inv, None]
        active = activation[use, block]
        exposure = cyclic_response(active, group_weights).round(7)
        for _ in range(MAX_TERMS_PER_BLOCK):
            contrast = (2 * residual[use].astype(float) - 1) * group_weights
            gain = cyclic_response(active, contrast).round(7)
            normalized = np.where(exposure > MIN_EXPOSURE, gain / np.maximum(exposure, 1), -1.)
            candidate = int(np.argmax(normalized))
            if normalized[candidate] < MIN_ADVANTAGE or (block, candidate) in shape:
                break
            alternate = np.sort(normalized)[-2]
            selected["accepted"].append({"offset": candidate, "signed_improvement": float(normalized[candidate]),
                                         "runner_up": float(alternate), "affected": float(exposure[candidate])})
            shape.append((block, candidate))
            residual ^= np.roll(activation[:, block], -candidate, axis=1)
    return shape, receipts


def totals(syndrome, mask, groups):
    bad = syndrome & mask
    by_group = []
    for group in np.unique(groups):
        use = groups == group
        n = int(mask[use].sum())
        if n:
            by_group.append({"group": int(group), "seen": n, "violations": int(bad[use].sum())})
    rates = [1 - row["violations"] / row["seen"] for row in by_group]
    return {"seen": int(mask.sum()), "violations": int(bad.sum()), "group_accuracy": float(np.mean(rates)) if rates else None,
            "groups": by_group}


def score_word_set(bits, mask, groups, sys, parity, reference, reference_seen):
    positions = np.column_stack((sys, parity))
    syndrome, seen = project_checks(bits, mask, positions)
    shifted = sys // LIFT * LIFT + (sys + 1) % LIFT
    wrong, wrong_seen = project_checks(bits, mask, np.column_stack((shifted, parity)))
    par_value, par_seen = project_checks(bits, mask, parity)
    for_comparison = seen & reference_seen
    common = seen & wrong_seen
    return {"with_systematic": totals(syndrome, seen, groups),
            "shared_with_padding_reference": totals(syndrome, for_comparison, groups),
            "padding_reference_only": totals(par_value ^ reference, for_comparison & par_seen, groups),
            "shared_with_shifted_systematic": totals(syndrome, common, groups),
            "shifted_systematic_one": totals(wrong, common, groups),
            "previously_uncertain_rotation": totals(syndrome, seen & ~reference_seen, groups)}


def packed_arrays(archive, pair):
    return tuple(np.unpackbits(archive[name], axis=1).astype(bool) for name in pair)


def held_only(bits, mask, rows):
    held = np.array([not row["calibration"] for row in rows])
    return bits[held], mask[held], np.array([row["group"] for row in rows])[held]


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    words = load(EXTRACTION, "E083", "mcs66_padding_words.json")
    edge = words["systematic_end"]
    n = words["codeword_bits"]
    rows = words["rows"]
    pn = equality_predictions(np.arange(edge), pn_bits(), words["pn_model"]).astype(bool)
    data = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    with np.load(data, allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ("bits", "mask"))
    training = np.array([row["calibration"] for row in rows])
    groups = np.array([row["group"] for row in rows])
    bounds = np.array([row["bound"] for row in rows])
    active = (bits[:, :edge] ^ pn) & mask[:, :edge] & (np.arange(edge) < bounds[:, None])
    active = active.reshape(len(bits), -1, LIFT)
    source = load("v080-sparse-qc-orbits", "E082", "sparse_qc_orbits.json")
    admitted = [arm for arm in source["outcomes"]["66"] if arm["length"] == LIFT]
    cohorts = load("v080-gain-padded-words", "E082", "gain_padded_words.json")["cohorts"]
    zipped = checked_artifact("v080-gain-padded-words", "E082", "gain_padded_words.npz", inputs)
    with np.load(zipped, allow_pickle=False) as z:
        non66 = held_only(*packed_arrays(z, ("nonterminal_66_bits", "nonterminal_66_seen")), cohorts["nonterminal_66"])
        non70 = held_only(*packed_arrays(z, ("nonterminal_70_bits", "nonterminal_70_seen")), cohorts["nonterminal_70"])
    raw_run = "v026-ut-qam-extract-repeat"
    raw_path = checked_artifact(raw_run, "E028", "ut_qam_symbols.npz", inputs)
    raw_meta = load(raw_run, "E028", "ut_qam_extraction.json")
    raw, _, _ = collect(raw_path, raw_meta)
    fit_result, matrices = [], []
    for arm in admitted:
        par = edge + orbit_positions(arm["shape"], LIFT)
        reference = np.array([bool(value) for value in arm["fixed_affine_constants"]])
        trusted = np.array([value is not None for value in arm["fixed_affine_constants"]])
        values, seen = project_checks(bits, mask, par)
        shape, trace = fit_taps(values ^ reference, seen & trusted, active, bounds, groups, training)
        sys_positions = orbit_positions(shape, LIFT)
        fill = np.bitwise_xor.reduce(pn[sys_positions], axis=-1)
        matrices.append(np.column_stack((sys_positions, par)))
        fine, base, _, records = raw[66]
        raw_groups = np.array([r["frame"] for r in records])
        raw_base = score_word_set(base.astype(bool), np.ones(base.shape, bool), raw_groups,
                                  sys_positions, par, reference, trusted)
        raw_fine = score_word_set(fine.astype(bool), np.ones(fine.shape, bool), raw_groups,
                                  sys_positions, par, reference, trusted)
        result = {"parity_shape": arm["shape"], "systematic_shape": shape, "fit_trace": trace,
                  "known_padding_constants": {"matched": int((trusted & (fill == reference)).sum()),
                                              "compared": int(trusted.sum())},
                  "raw_250_baseline": raw_base, "raw_250_refined": raw_fine,
                  "archive_terminal_held": score_word_set(bits[~training], mask[~training], groups[~training],
                                                          sys_positions, par, reference, trusted),
                  "archive_nonterminal_held": score_word_set(*non66, sys_positions, par, reference, trusted),
                  "wrong_format_mcs70": score_word_set(*non70, sys_positions, par, reference, trusted)}
        fit_result.append(result)
        print("Recovered", shape, "known PN", result["known_padding_constants"], flush=True)
    indices = np.vstack(matrices)
    if any(len(set(row)) != len(row) for row in indices):
        raise ValueError("Duplicate coordinate in a candidate check row")
    checks = np.zeros((len(indices), n), dtype=np.uint8)
    checks[np.arange(len(indices))[:, None], indices] = 1
    report = {"schema": "starlink-mcs66-recovered-systematic-checks/v1", "lift": LIFT,
              "rank": binary_rank(checks), "shape": list(checks.shape),
              "candidate_systematic_end": edge, "results": fit_result,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E083")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs66_partial_checks.npz", row_support=indices, check_matrix_packed=np.packbits(checks, axis=1))
    (out / "mcs66_systematic_checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Partial GF(2) rank", report["rank"])


if __name__ == "__main__":
    main()
