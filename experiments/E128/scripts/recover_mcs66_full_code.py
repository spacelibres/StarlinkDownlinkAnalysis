"""Recover the remaining QC checks using the active first systematic block."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH, paired_syndromes
from audit_cross_rate_checks import shift_systematic
from audit_qam_pn15 import equality_predictions
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from extend_mcs66_parity_matrix import exact_completions, padding_constant
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, LIFT, fit_taps, packed_arrays, project_checks, totals
from search_mcs66_early_projections import PAIR_BITS, PAIR_ID_BITS, matching_pairs, signatures_for
from sparse_code5_core import binary_rank


SYSTEMATIC = 12288
SEED = 20261121128
WIDTHS = (16, 32, 64)
DIFFERENCES = 24


def parity_blocks(pivot):
    return (pivot, *(b for b in range(4, 8) if b != 4 + (pivot - 1) % 4))


def nominate(bits, visible, pivot):
    coords, signatures, _ = signatures_for(bits, visible, 0)
    blocks = coords // LIFT
    anchor = np.flatnonzero(coords == (pivot + 1) * LIFT)
    if not len(anchor):
        return set()
    left, middle, right = (np.flatnonzero(blocks == b + 1) for b in parity_blocks(pivot)[1:])
    if not len(right):
        return set()
    i, j = np.meshgrid(left, middle, indexing="ij")
    key = signatures[i] ^ signatures[j]
    pairs = ((key << PAIR_BITS) | (i.astype(np.uint64) << PAIR_ID_BITS) | j.astype(np.uint64)).ravel()
    pairs.sort()
    early = np.r_[np.flatnonzero(blocks == 0), len(coords)]
    padded = np.r_[signatures, np.uint64(0)]
    positions = np.r_[coords, -1]
    target = (padded[early, None] ^ signatures[right][None, :] ^ signatures[anchor[0]]).ravel()
    hits = set()
    for match, a, b in matching_pairs(pairs, pairs >> PAIR_BITS, target, np.uint64(0)):
        s, k = divmod(match, len(right))
        # 512 denotes an absent first-block tap. Keep the tuple in 37 bits.
        first = int(positions[early[s]])
        first = LIFT if first < 0 else first
        x, y, z = (int(positions[t] % LIFT) for t in (a, b, right[k]))
        hits.add((((first * LIFT + x) * LIFT + y) * LIFT) + z)
    return hits


def unpack_shape(packed):
    first, rest = divmod(packed, LIFT ** 3)
    a, final = divmod(rest, LIFT ** 2)
    b, c = divmod(final, LIFT)
    return [-1 if first == LIFT else first, a, b, c]


def search(bits, visible, short):
    votes = [Counter() for _ in range(4)]
    trials = []
    rng = np.random.default_rng(SEED)
    coordinates = np.arange(bits.shape[1])
    for width in WIDTHS:
        for start in range(0, LIFT, width):
            take = rng.permutation(short)
            rest = np.resize(take[1:], DIFFERENCES)
            phase = rng.integers(start, start + width, size=DIFFERENCES)
            columns = coordinates // LIFT * LIFT + (coordinates + phase[:, None]) % LIFT
            diff = bits[take[0], columns] ^ bits[rest[:, None], columns]
            mask = visible[take[0], columns] & visible[rest[:, None], columns]
            counts = []
            for pivot, counter in enumerate(votes):
                candidates = nominate(diff, mask, pivot)
                counter.update(candidates)
                counts.append(len(candidates))
            trials.append({"window": [start, start + width], "reference": int(take[0]), "others": rest.tolist(),
                           "rotations": phase.tolist(), "candidate_counts": counts})
    return votes, trials


def assess(bits, seen, groups, support):
    s, o = project_checks(bits, seen, support)
    return totals(s, o, groups)


def complete(pivot, shape, bits, masks, active, active_seen, pn, fit, groups, bounds, ordinary):
    first, *offsets = shape
    early = [(0, first)] if first >= 0 else []
    parity_shape = list(zip(parity_blocks(pivot), (0, *offsets), strict=True))
    parity = SYSTEMATIC + orbit_positions(parity_shape, LIFT)
    value, observed = project_checks(bits, masks, parity)
    ref, good = padding_constant(value, observed, active, active_seen, early, groups, fit & (bounds <= LIFT))
    residual = value ^ ref
    if early:
        a, m = project_checks(active, active_seen, orbit_positions(early, LIFT))
        residual ^= a
        observed &= m
    rest = active.copy()
    rest[:, :LIFT] = False
    taps, trace = fit_taps(residual, observed & good, rest.reshape(len(bits), -1, LIFT), bounds, groups, fit)
    taps = early + taps
    if good.sum() < 256 or not taps:
        raise ValueError("Too few trusted padding rotations")
    learned = np.bitwise_xor.reduce(pn[orbit_positions(taps, LIFT)], axis=-1)
    dictionary = pn[orbit_positions([divmod(i, LIFT) for i in range(SYSTEMATIC)], LIFT)].T
    repairs = exact_completions(dictionary[:, good], (learned ^ ref)[good])
    ranked = []
    for repair in repairs:
        corrected = sorted(set(taps) ^ set(divmod(i, LIFT) for i in repair))
        h = np.column_stack((orbit_positions(corrected, LIFT), parity))
        score = assess(*ordinary, h)
        ranked.append((corrected, h, {"toggled_terms": [divmod(i, LIFT) for i in repair],
                                      "systematic_terms": corrected, "calibration": score}))
    ranked.sort(key=lambda item: -item[-1]["calibration"]["group_accuracy"])
    if not ranked or ranked[0][-1]["calibration"]["group_accuracy"] < .75:
        raise ValueError("No calibrated completion of nominated shape")
    return ranked[0][1], {"pivot": pivot, "parity_shape": parity_shape, "early_systematic": early,
                           "unrepaired_systematic": taps, "trusted_rotations": int(good.sum()),
                           "systematic_fit": trace, "calibration_completions": [r[2] for r in ranked]}


def prepare_middle(artifact, excluded):
    name = "v125-cross-rate-checks"
    state = read(artifact(name, "E127", "cross_rate_checks.json"))
    result = {}
    with np.load(artifact(name, "E127", "cross_rate_middle_words.npz"), allow_pickle=False) as data:
        for code in (66, 85, 70):
            cohort = state["archive_middle_words"][str(code)]
            hold = np.array([not row["calibration"] and row["group"] not in excluded for row in cohort])
            bits, seen = packed_arrays(data, (f"mcs{code}_bits", f"mcs{code}_seen"))
            result[code] = bits[hold], seen[hold], np.array([row["group"] for row in cohort])[hold]
    return result


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    meta = read(art(EXTRACTION, "E083", "mcs66_padding_words.json"))
    with np.load(art(EXTRACTION, "E083", "mcs66_padding_words.npz"), allow_pickle=False) as z:
        bits, masks = packed_arrays(z, ("bits", "mask"))
    with np.load(art("v125-cross-rate-checks", "E127", "mcs66_six_families.npz"), allow_pickle=False) as z:
        old = z["row_support"]
    groups = np.array([r["group"] for r in meta["rows"]])
    fit = np.array([r["calibration"] for r in meta["rows"]])
    bounds = np.array([r["bound"] for r in meta["rows"]])
    short = np.flatnonzero(fit & (bounds <= LIFT) & (masks[:, SYSTEMATIC:].mean(axis=1) >= .98))
    if len(np.unique(groups[short])) != len(short):
        raise ValueError("Registration rows must use distinct calibration groups")
    pn = equality_predictions(np.arange(SYSTEMATIC), pn_bits(), meta["pn_model"]).astype(bool)
    possible = np.arange(SYSTEMATIC) < bounds[:, None]
    active = (bits[:, :SYSTEMATIC] ^ pn) & masks[:, :SYSTEMATIC] & possible
    active_seen = masks[:, :SYSTEMATIC] | ~possible
    register = np.column_stack((active[:, :LIFT], bits[:, SYSTEMATIC:]))
    register_seen = np.column_stack((active_seen[:, :LIFT], masks[:, SYSTEMATIC:]))
    votes, trials = search(register, register_seen, short)
    tops = [[{"shape": unpack_shape(word), "nominations": count}
             for word, count in sorted(counter.items(), key=lambda r: (-r[1], r[0]))[:10]] for counter in votes]
    for pivot, family, fixed_first in ((1, 4, -1), (2, 5, 373)):
        expected = [fixed_first, *(int(x % LIFT) for x in old[family * LIFT] if x >= SYSTEMATIC + 4 * LIFT)]
        if tops[pivot][0]["shape"] != expected:
            raise ValueError("Known four-parity family did not win its positive control")
    for i in (0, 3):
        if tops[i][0]["nominations"] <= max(2, tops[i][1]["nominations"]):
            raise ValueError("New motif has no uniquely recurring candidate")
    print("Frozen nominees", [row[0] for row in tops], flush=True)
    run = "v080-gain-padded-words"
    cohorts = read(art(run, "E082", "gain_padded_words.json"))["cohorts"]["nonterminal_66"]
    with np.load(art(run, "E082", "gain_padded_words.npz"), allow_pickle=False) as z:
        b, m = packed_arrays(z, ("nonterminal_66_bits", "nonterminal_66_seen"))
    training = np.array([r["calibration"] for r in cohorts])
    ordinary = b[training], m[training], np.array([r["group"] for r in cohorts])[training]
    excluded = read(art("v122-mcs85-archive-records-verified", "E124", "mcs85_archive_records.json"))[
        "archive_overlapping_groups_excluded"]
    middle = prepare_middle(art, excluded)
    additions, receipts = [], []
    for i in (0, 3):
        h, result = complete(i, tops[i][0]["shape"], bits, masks, active, active_seen, pn, fit, groups, bounds, ordinary)
        control = shift_systematic(h, SYSTEMATIC)
        result["held_middle"] = {code: dict(zip(("reference", "shifted"), paired_syndromes(*data, h, control),
                                               strict=True)) for code, data in middle.items()}
        for code, gate in ((66, .75), (85, .58)):
            check = result["held_middle"][code]
            if check["reference"]["group_accuracy"] < gate or check["shifted"]["group_accuracy"] > .55:
                raise ValueError(f"Held MCS {code} does not verify a new family")
        additions.append(h)
        receipts.append(result)
    support = np.sort(np.vstack((old, *additions)), axis=1)
    expanded = np.zeros((len(support), BIT_LENGTH), dtype=np.uint8)
    expanded[np.arange(len(support))[:, None], support] = 1
    rank = binary_rank(expanded)
    parity_rank = binary_rank(expanded[:, SYSTEMATIC:].copy())
    if rank != BIT_LENGTH - SYSTEMATIC or parity_rank != rank:
        raise ValueError("The full candidate has a rank-deficient parity projection")
    detail = {"schema": "starlink-mcs66-full-code-recovery/v1", "seed": SEED, "systematic_bits": SYSTEMATIC,
              "codeword_bits": BIT_LENGTH, "rank": rank, "parity_rank": parity_rank,
              "short_rows": short.tolist(), "short_groups": groups[short].tolist(), "trials": trials,
              "top_nominations": tops, "distinct_nominations": [len(v) for v in votes], "recovered": receipts,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    path = experiment_output("E128")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "mcs66_full_checks.npz", row_support=support,
                        check_matrix_packed=np.packbits(expanded, axis=1))
    (path / "mcs66_full_recovery.json").write_text(json.dumps(detail, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    print("Recovered check and parity ranks", rank, parity_rank)


if __name__ == "__main__":
    main()
