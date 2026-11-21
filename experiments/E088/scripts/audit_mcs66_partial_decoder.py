"""Measure partial MCS 66 decoding against held, known reference bits."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_mcs66_single_erasures import repeat_partners
from audit_qam_pn15 import collect, equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import EXTRACTION, LIFT, packed_arrays


GUARD_BITS = 128
EDGE_CARRIER = 509
NORMALIZATION = .8
CHECKPOINTS = (0, 1, 3, 6)
RAW_VARIANCE = .03
ZERO_THRESHOLD = 1e-8
BOOTSTRAP_SEED = 2026112189


def constellation_llr(samples, points, mapping, noise, log_map=True):
    likelihood = -(abs(samples[..., None] - points) ** 2) / noise
    reduction = logsumexp if log_map else np.max
    by_lane = [reduction(likelihood[..., ~lane], axis=-1) - reduction(likelihood[..., lane], axis=-1)
               for lane in mapping.astype(bool).T]
    return np.stack(by_lane, axis=-1)


def syndrome_count(bits, support, visible=True):
    return (np.bitwise_xor.reduce(bits[:, support], axis=-1) & visible).sum(axis=-1)


def min_sum(prior, support, normalization, iterations):
    if iterations == 0:
        return prior.copy()
    columns = support.ravel()
    incidence = csr_matrix((np.ones(len(columns), dtype=prior.dtype), (columns, np.arange(len(columns)))),
                           shape=(prior.shape[1], len(columns)))
    posterior = prior.copy()
    for start in range(0, len(prior), 8):
        original = prior[start:start + 8]
        q = original[:, support]
        for _ in range(iterations):
            absolute = abs(q)
            minima = np.partition(absolute, 1, axis=-1)[..., :2]
            extrinsic = np.where(absolute == minima[..., :1], minima[..., 1:2], minima[..., :1])
            negative = q < 0
            check_sign = np.bitwise_xor.reduce(negative, axis=-1)[..., None]
            sent = normalization * np.where(check_sign ^ negative, -extrinsic, extrinsic)
            full = original + (incidence @ sent.reshape(len(original), -1).T).T
            q = full[:, support] - sent
        posterior[start:start + len(original)] = full
    return posterior


def output_decisions(posterior, bits, seen):
    resolved = abs(posterior) > ZERO_THRESHOLD
    return np.where(resolved, posterior < 0, bits & seen), resolved | seen


def archive_prior(bits, seen, rows, reference, bounds, fit):
    indices = np.arange(bits.shape[1])
    starts = np.array([r["resource"] for r in rows])
    carriers = NONPILOT[(starts[:, None] + indices // 4) % len(NONPILOT)]
    classes = (abs(carriers) >= EDGE_CARRIER).astype(np.uint8) * 4 + indices % 4
    eligible = fit[:, None] & seen & (indices >= bounds[:, None] + GUARD_BITS) & (indices < len(reference))
    truth = np.pad(reference, (0, bits.shape[1] - len(reference)))
    counts = np.bincount(classes[eligible], minlength=8)
    errors = np.bincount(classes[eligible & (bits != truth)], minlength=8)
    rates = (errors + .5) / (counts + 1)
    if not counts.all() or (rates >= .5).any():
        raise ValueError("Archive reliability classes could not be calibrated")
    llr = np.log((1 - rates) / rates).astype(np.float32)
    prior = (1 - 2 * bits.astype(np.float32)) * llr[classes] * seen
    return prior, {"counts": counts.tolist(), "errors": errors.tolist(), "llr": llr.tolist()}


def pn_metrics(bits, seen, originally_seen, reference, region, groups):
    mismatch = bits != reference
    retain = region & originally_seen
    filled = region & seen & ~originally_seen
    rows = []
    for group in sorted(set(groups)):
        take = groups == group
        rows.append({
            "group": int(group), "originally_seen": int(retain[take].sum()),
            "observed_errors": int((retain[take] & mismatch[take]).sum()),
            "proposed": int(filled[take].sum()),
            "proposal_errors": int((filled[take] & mismatch[take]).sum()),
        })
    result = {key: sum(row[key] for row in rows)
              for key in ("originally_seen", "observed_errors", "proposed", "proposal_errors")}
    rates = [row["observed_errors"] / row["originally_seen"] for row in rows if row["originally_seen"]]
    return {**result, "group_error_rate": float(np.mean(rates)), "groups": rows}


def repeated_word_metrics(bits, seen, original, original_seen, links, fit, pn_regions):
    results = []
    for i, partners in sorted(links.items()):
        for j in sorted(partners):
            if fit[i] or fit[j] or i == j:
                continue
            common = original_seen[i] & original_seen[j]
            new = ~original_seen[i] & seen[i] & original_seen[j]
            both_pn = pn_regions[i] & pn_regions[j]
            hard_mismatch = (original[i] != original[j]) & common
            partner_mismatch = (bits[i] != original[j]) & common
            mutual_mismatch = (bits[i] != bits[j]) & common
            results.append({
                "source": i, "partner": j, "originally_shared": int(common.sum()),
                "shared_pn": int((common & both_pn).sum()),
                "hard_disagreements": int(hard_mismatch.sum()),
                "hard_disagreements_in_shared_pn": int((hard_mismatch & both_pn).sum()),
                "decoded_disagreements": int(partner_mismatch.sum()),
                "decoded_disagreements_in_shared_pn": int((partner_mismatch & both_pn).sum()),
                "mutual_decoded_disagreements": int(mutual_mismatch.sum()),
                "mutual_decoded_disagreements_in_shared_pn": int((mutual_mismatch & both_pn).sum()),
                "proposed": int(new.sum()),
                "proposal_disagreements": int(((bits[i] != original[j]) & new).sum()),
            })
    return results


def grouped_improvement(baseline, scored):
    groups = [(a["observed_errors"] - b["observed_errors"]) / a["originally_seen"]
              for a, b in zip(baseline["groups"], scored["groups"], strict=True) if a["originally_seen"]]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(groups, (2000, len(groups)), replace=True).mean(axis=1)
    return {"mean_baseline_minus_decoded": float(np.mean(groups)),
            "bootstrap_percentile_interval": np.percentile(samples, [2.5, 97.5]).tolist(),
            "groups": len(groups)}


def compare_pn_proposals(baseline, row_ids, bits, visible, original_seen, reference, regions):
    """Compare PN correctness on identical single-erasure and iterative proposals."""
    shared = {"positions": 0, "single_erasure_errors": 0, "iterative_errors": 0}
    lookup = {row: i for i, row in enumerate(row_ids)}
    for proposal in baseline:
        i = lookup.get(proposal["row"])
        if i is None:
            continue
        for pos, value, _ in proposal["candidates"]:
            if pos >= len(reference) or not regions[i, pos] or not visible[i, pos]:
                continue
            if original_seen[i, pos]:
                raise ValueError("Single-erasure proposal overlaps an original observation")
            shared["positions"] += 1
            shared["single_erasure_errors"] += int(value != reference[pos])
            shared["iterative_errors"] += int(bits[i, pos] != reference[pos])
    return shared


def raw_scores(inputs, matrices, chosen, archive_rows, reference):
    run = "v026-ut-qam-extract-repeat"
    source = checked_artifact(run, "E028", "ut_qam_symbols.npz", inputs)
    raw_meta = read(checked_artifact(run, "E028", "ut_qam_extraction.json", inputs))
    extraction, points, mapping = collect(source, raw_meta)
    bits, _, values, rows = extraction[66]
    index = read(checked_artifact("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json", inputs))["overlap"]
    by_candidate = {r["ut_candidate"]: r["qin_frame"] for r in index["pairs"]}
    by_archive = {(r["frame"], r["resource"]): r for r in archive_rows}
    matched = []
    for witness in rows:
        key = (by_candidate.get(witness["frame"]), witness["resource_start"])
        if witness["last_word_of_allocation"] and key in by_archive:
            donor = by_archive[key]
            matched.append({**witness, "archive_frame": donor["frame"], "bound": donor["bound"],
                            "archive_calibration": donor["calibration"]})
    take = [r["candidate_index"] for r in matched]
    word = bits[take].astype(bool)
    prior = constellation_llr(values[take], points, mapping, RAW_VARIANCE).reshape(word.shape).astype(np.float32)
    valid = np.ones_like(word)
    bound = np.array([r["bound"] for r in matched])
    regions = np.arange(len(reference)) >= bound[:, None] + GUARD_BITS
    totals = {}
    settings = {"nearest": word, "llr_only": output_decisions(prior, word, valid)[0]}
    for name, rows_check in matrices.items():
        post = min_sum(prior, rows_check, NORMALIZATION, chosen["five"] if name == "shifted" else chosen[name])
        settings[name] = output_decisions(post, word, valid)[0]
    for name, result in settings.items():
        totals[name] = [int(((row[:len(reference)] != reference) & region).sum())
                        for row, region in zip(result, regions, strict=True)]
    return {"word_observations": matched, "scored_positions": regions.sum(axis=1).tolist(), "errors": totals}


def main():
    require_run_context()
    inputs = {}
    source = checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.npz", inputs)
    words = read(checked_artifact(EXTRACTION, "E083", "mcs66_padding_words.json", inputs))
    with np.load(source, allow_pickle=False) as z:
        original, seen = packed_arrays(z, ("bits", "mask"))
    rows = words["rows"]
    fit = np.array([r["calibration"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    bounds = np.array([r["bound"] for r in rows])
    edge = words["systematic_end"]
    reference = equality_predictions(np.arange(edge), pn_bits(), words["pn_model"]).astype(bool)
    path = checked_artifact("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz", inputs)
    with np.load(path, allow_pickle=False) as z:
        five = z["row_support"]
    path = checked_artifact("v082-mcs66-expanded-checks", "E084", "mcs66_expanded_checks.npz", inputs)
    with np.load(path, allow_pickle=False) as z:
        four = z["row_support"]
    if not np.array_equal(five[:len(four)], four):
        raise ValueError("The ablation matrices do not share the first four families")
    shifted = np.where(five < edge, five // LIFT * LIFT + (five + 1) % LIFT, five)
    matrices = {"four": four, "five": five, "shifted": shifted}
    prior, reliability = archive_prior(original, seen, rows, reference, bounds, fit)
    region = np.arange(edge) >= bounds[:, None] + GUARD_BITS
    full_pn = np.pad(region, ((0, 0), (0, original.shape[1] - edge)))
    repeats = read(checked_artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json", inputs))
    links = repeat_partners(repeats, rows)
    single_erasures = read(checked_artifact("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.json", inputs))

    calibration = {name: {} for name in ("four", "five")}
    for name in calibration:
        for it in CHECKPOINTS:
            post = min_sum(prior[fit], matrices[name], NORMALIZATION, it)
            possible, visible = output_decisions(post, original[fit], seen[fit])
            calibration[name][it] = pn_metrics(possible[:, :edge], visible[:, :edge], seen[fit, :edge],
                                               reference, region[fit], groups[fit])
    chosen = {name: min(curve, key=lambda it: (curve[it]["group_error_rate"], it))
              for name, curve in calibration.items()}

    held_bits, held_seen = original[~fit], seen[~fit]
    base = pn_metrics(held_bits[:, :edge], held_seen[:, :edge], held_seen[:, :edge],
                      reference, region[~fit], groups[~fit])
    valid_rows = np.flatnonzero(~fit)
    holds, packed = {}, {"source_row_indices": valid_rows}
    held_links = repeated_word_metrics(original, seen, original, seen, links, fit, full_pn)
    mapped = {int(row): i for i, row in enumerate(valid_rows)}
    local_links = {mapped[i]: {mapped[j] for j in partners if j in mapped}
                   for i, partners in links.items() if i in mapped}
    true_check_seen = held_seen[:, five].all(axis=-1)
    baseline_syndrome = int(syndrome_count(held_bits, five, true_check_seen).sum())
    for name, support in matrices.items():
        # Calibration uses only the score above. Apply each frozen setting once.
        it = chosen["five"] if name == "shifted" else chosen[name]
        post = min_sum(prior[~fit], support, NORMALIZATION, it)
        bits, visible = output_decisions(post, held_bits, held_seen)
        result = pn_metrics(bits[:, :edge], visible[:, :edge], held_seen[:, :edge], reference, region[~fit], groups[~fit])
        repeated = repeated_word_metrics(bits, visible, held_bits, held_seen, local_links, fit[~fit], full_pn[~fit])
        for item in repeated:
            item["source"] = int(valid_rows[item["source"]])
            item["partner"] = int(valid_rows[item["partner"]])
        holds[name] = {"iterations": it, "pn": result, "delta": grouped_improvement(base, result),
                       "repeat_partner_diagnostics": repeated,
                       "nonzero_syndromes_on_originally_visible_true_checks": int(syndrome_count(bits, five, true_check_seen).sum()),
                       "originally_visible_true_checks": int(true_check_seen.sum())}
        holds[name]["shared_pn_proposals"] = compare_pn_proposals(
            single_erasures["single_erasure_proposals"], valid_rows, bits, visible, held_seen, reference, region[~fit]
        )
        packed[name + "_decision_bits"] = np.packbits(bits & visible, axis=1)
        packed[name + "_decision_mask"] = np.packbits(visible, axis=1)

    result = {"schema": "starlink-mcs66-partial-decoder/v1", "guard_bits": GUARD_BITS,
              "edge_carrier_threshold": EDGE_CARRIER, "normalization": NORMALIZATION,
              "bootstrapping_seed": BOOTSTRAP_SEED, "raw_variance": RAW_VARIANCE,
              "archive_calibration_reliability": reliability, "calibration_curves": calibration,
              "selected_iterations": chosen, "held_baseline_pn": base, "held_baseline_pairs": held_links,
              "held_baseline_nonzero_syndromes": baseline_syndrome,
              "archive_held": holds, "raw_iq": raw_scores(inputs, matrices, chosen, rows, reference),
              "input_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()}}
    out = experiment_output("E088")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs66_partial_decoder.npz", **packed)
    (out / "mcs66_partial_decoder.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Iterations", chosen, "held PN errors", base["observed_errors"],
          {name: value["pn"]["observed_errors"] for name, value in holds.items()},
          "raw errors", result["raw_iq"]["errors"])


if __name__ == "__main__":
    main()
