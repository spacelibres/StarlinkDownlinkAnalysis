"""Test withheld Code 5 parity observations and attempt full-word decoding."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_offset_h2_systematics import HALF_WORD, WORD_SIZE, orient_h1
from audit_ut_offset_grid_pilots import HEADER_LENGTH
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from sparse_code5_core import normalized_min_sum


SEED = 2026112146
LIFT = 128
GROUP_COUNT = HALF_WORD//LIFT
RHO = .6
PRIOR_WEIGHTS = (0., .3, .6, .9)
OFFSETS = (1, -1, 17, -17, 64, -64, 128, -128, 256)
STEPS = 60


def cosine(a, b):
    return float((a*b).sum()/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))


def rms_scaled(values, visible):
    return values/np.maximum(np.sqrt(np.mean(values[..., visible]**2, axis=-1, keepdims=True)), 1e-30)


def model_scale(strength):
    mu = np.clip(strength, 0., .95)
    return mu/(1-mu**2)


def log_odds(mean):
    return np.arctanh(np.clip(mean, -1+1e-12, 1-1e-12))


def prior_for(words, gain):
    sys = words[..., :HALF_WORD]
    diag = rms_scaled(sys, slice(None))
    return np.tanh(np.sum(gain[:, None, None]*diag, axis=0))


def targeted_checks(columns, hidden):
    masked = hidden[columns]
    selected = columns[masked.sum(axis=1) == 1]
    held = hidden[selected]
    target = selected[held]-HALF_WORD
    context = selected[~held].reshape(len(selected), columns.shape[1]-1)
    return target, context


def predict(words, gain, prior, columns, hold_groups):
    shape = (*words.shape[:-1], HALF_WORD)
    prediction = np.zeros(shape)
    scope = np.zeros(HALF_WORD, dtype=bool)
    control = np.setdiff1d(np.arange(GROUP_COUNT), hold_groups)
    parity = np.arange(HALF_WORD)//LIFT
    for subset in (hold_groups, control):
        hidden = np.r_[np.zeros(HALF_WORD, dtype=bool), np.isin(parity, subset)]
        target, context = targeted_checks(columns, hidden)
        # Slice before applying any scale or prior. This keeps the prediction
        # unchanged even if arbitrary data replace the whole masked half.
        visible = np.flatnonzero(~hidden)
        mapping = np.full(WORD_SIZE, -1, dtype=int)
        mapping[visible] = np.arange(len(visible))
        z = words[..., visible]
        likelihood = gain[:, None, None]*rms_scaled(z, slice(None))
        likelihood[..., :HALF_WORD] += log_odds(prior)
        local = np.tanh(likelihood)
        extrinsic = log_odds(np.prod(local[..., mapping[context]], axis=-1))
        merged = np.zeros((len(target), HALF_WORD))
        merged[np.arange(len(target)), target] = 1
        count = np.maximum(merged.sum(axis=0), 1)
        prediction += np.tanh(extrinsic @ merged/count)
        scope[target] = True
    return prediction, scope


def rotations(predicted, observed):
    blocks = observed.reshape(*observed.shape[:-1], GROUP_COUNT, LIFT)
    return [cosine(predicted, np.roll(blocks, shift, axis=-1).reshape(observed.shape))
            for shift in range(1, LIFT)]


def check_row_overlap(original, shifted):
    base = set(map(tuple, original))
    moved = set(map(tuple, np.sort(shifted, axis=-1)))
    return len(base & moved)


def prediction_group(words, gain, prior, pn_prior, columns, hold, ticks):
    observed = rms_scaled(words[..., HALF_WORD:], slice(None))
    arms = [(f"repeat_{rho:g}", rho*prior) for rho in PRIOR_WEIGHTS]
    arms += [("wrong_prior_code5_block", RHO*np.roll(prior, LIFT, axis=-1)), ("pn_only", pn_prior)]
    scores = {}
    for name, external in arms:
        predicted, scope = predict(words, gain, external, columns, hold)
        scores[name] = {"target_columns": int(scope.sum()), "score": cosine(predicted, observed),
                        "score_by_tick": {str(t): cosine(a, b) for t, a, b in zip(ticks, predicted, observed, strict=True)},
                        "rotation_control_scores": rotations(predicted, observed)}
        if name == "repeat_0.6":
            primary, primary_scope = predicted, scope
    comparisons = []
    for offset in OFFSETS:
        wrong = (columns+offset) % WORD_SIZE
        prediction, support = predict(words, gain, RHO*prior, wrong, hold)
        common = primary_scope & support
        comparisons.append({"offset": offset, "shared_check_rows": check_row_overlap(columns, wrong),
                            "common_target_columns": int(common.sum()),
                            "nominal_score_on_common_columns": cosine(primary[..., common], observed[..., common]),
                            "shifted_score_on_common_columns": cosine(prediction[..., common], observed[..., common])})
    return {"arms": scores, "column_shift_controls": comparisons}, primary, observed


def sum_product(llr, columns):
    """Damped parallel SPA for the frozen regular-weight check rows."""
    edge = llr[columns].copy()
    leave_one = np.array([[k for k in range(columns.shape[1]) if k != j]
                          for j in range(columns.shape[1])])
    bits = llr < 0
    failures = int(np.count_nonzero(bits[columns].sum(axis=1) % 2))
    step = 0
    for step in range(1, STEPS+1):
        if failures == 0:
            step -= 1
            break
        outgoing = 2*log_odds(np.prod(np.tanh(edge/2)[:, leave_one], axis=-1))
        posterior = llr+np.bincount(columns.ravel(), weights=outgoing.ravel(), minlength=WORD_SIZE)
        bits = posterior < 0
        failures = int(np.count_nonzero(bits[columns].sum(axis=1) % 2))
        edge = .5*edge+.5*(posterior[columns]-outgoing)
    return {"bits": bits, "converged": failures == 0, "iterations": step, "final_failed_checks": failures}


def decode_range(words, gain, prior, checks, columns, generator, ticks, truth=None):
    soft = gain[:, None, None]*rms_scaled(words, slice(None))
    output = []
    for rho in (0., RHO):
        profile = soft.copy()
        profile[..., :HALF_WORD] += log_odds(rho*prior)
        for i, j in np.ndindex(profile.shape[:2]):
            values = 2*profile[i, j]
            paths = {"min_sum": normalized_min_sum(values, checks, max_iterations=STEPS),
                     "sum_product": sum_product(values, columns)}
            for name, row in paths.items():
                bits = row.pop("bits")
                predicted = (bits[:HALF_WORD].astype(np.uint8) @ generator) % 2
                result = {"tick": ticks[i], "word_index": j, "prior_rho": rho, "decoder": name,
                          "reencoding_errors": int(np.count_nonzero(predicted != bits[HALF_WORD:])), **row}
                if truth is not None:
                    result["truth_bit_errors"] = int(np.count_nonzero(bits != truth[i, j]))
                output.append(result)
    return output


def main():
    require_run_context()
    inputs = {}
    def artifact(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)
    e044 = "v042-ut-offset-grid-header"
    config = read(artifact(e044, "E044", "ut_offset_grid_pilots.json"))
    captured = artifact(e044, "E044", "ut_offset_grid_soft.npz")
    layout = read(artifact("v043-ut-offset-h2-final", "E045", "ut_offset_h2_systematics.json"))
    prior_data = layout["allocation_test"]
    with np.load(artifact("v011-sparse-code5", "E013", "sparse_code5_arrays.npz"), allow_pickle=False) as entry:
        checks = entry["check_matrix"]
    with np.load(artifact("v010-code5-parity-02", "E012", "code5_observations.npz"), allow_pickle=False) as entry:
        generator = entry["parity_generator"]
    if not np.all(checks.sum(axis=1) == 6):
        raise ValueError("Unexpected Code 5 check weight")
    columns = np.nonzero(checks)[1].reshape(len(checks), 6)
    control_path = artifact("v022-ut-replay", "E014", "ut_observations.npz")
    with np.load(control_path, allow_pickle=False) as entry:
        strong = entry["observed_code5_symbols"].real[:, None, :]
        exact = entry["decoded_code5_words"]
    rng = np.random.default_rng(SEED)
    hold = rng.permutation(GROUP_COUNT)[:4]
    with np.load(captured, allow_pickle=False) as entry:
        pools = {"eight": entry["fold_0_held"], "seven": entry["fold_1_held"], "overlap": entry["fold_0_overlap"]}
    ticks = {"eight": config["folds"][0]["held_ticks"], "seven": config["folds"][1]["held_ticks"],
             "overlap": config["overlap_ticks"]}
    vectors, gains = {}, {}
    for name, data in pools.items():
        aligned, strength = orient_h1(data, layout["h1_words_hex"])
        vectors[name] = aligned[:, HEADER_LENGTH:layout["h2_start_resource"]].real.reshape(len(aligned), -1, WORD_SIZE)
        gains[name] = model_scale(strength)
    pn = 1-2*equality_predictions(np.arange(HALF_WORD), pn_bits(), prior_data["pn_model"]).astype(float)
    report, arrays = {}, {}
    for train, test in (("eight", "seven"), ("seven", "eight"), ("pooled_main", "overlap")):
        if train == "pooled_main":
            priors = prior_for(np.concatenate([vectors[g] for g in ("eight", "seven")]),
                               np.concatenate([gains[g] for g in ("eight", "seven")]))
        else:
            priors = prior_for(vectors[train], gains[train])
        pn_prior = np.zeros(priors.shape)
        source_groups = ("eight", "seven") if train == "pooled_main" else (train,)
        for index in range(len(pn_prior)):
            rows = [row for row in prior_data["pn_support"] if row["train"] in source_groups
                    and row["word_index"] == index and row["half"] == 0]
            start, stop = max(row["start"] for row in rows), min(row["stop"] for row in rows)
            pn_prior[index, start:stop] = RHO*pn[start:stop]
        result, prediction, observed = prediction_group(vectors[test], gains[test], priors, pn_prior, columns, hold, ticks[test])
        report[test] = {"training_group": train, "held_ticks": ticks[test], **result,
                        "word_decoders": decode_range(vectors[test], gains[test], priors, checks, columns, generator, ticks[test])}
        arrays.update({f"{test}_extrinsic": prediction, f"{test}_observed": observed,
                       f"{test}_systematic_prior": priors})
        print(test, "held score", result["arms"]["repeat_0.6"]["score"], flush=True)
    strong_pred, _ = predict(strong, np.full(len(strong), model_scale(.90)), 0., columns, hold)
    measured_control = {"source": "E014 exact Code 5 positive control", "words": len(strong),
                        "predicted_to_measured": cosine(strong_pred, strong[..., HALF_WORD:]),
                        "predicted_to_decoded_sign": cosine(strong_pred, 1-2*exact[:, None, HALF_WORD:].astype(float))}
    synthetic = {}
    for mu in (.35, .85):
        sign = 1-2*exact[:16, None].astype(float)
        generated = mu*sign + np.sqrt(1-mu**2)*rng.standard_normal(sign.shape)
        decoded = decode_range(generated, np.full(len(generated), model_scale(mu)), 0., checks, columns,
                               generator, list(range(len(generated))), truth=exact[:16, None])
        synthetic[str(mu)] = {"reference_words": len(generated), "records_without_prior": [r for r in decoded if r["prior_rho"] == 0.]}
    output = experiment_output("E046")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "starlink-ut-offset-code5-parity/v1",
               "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
               "seed": SEED, "half_word_lift": LIFT, "parity_complement_groups": hold.tolist(),
               "priors": list(PRIOR_WEIGHTS), "primary_prior_rho": RHO, "maximum_iterations": STEPS,
               "measured_positive_control": measured_control, "results": report,
               "scope": "This is a fixed-code extrinsic parity test and conditional decoding attempt in one RF recording."}
    (output/"ut_offset_code5_parity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output/"synthetic_code5_controls.json").write_text(json.dumps(synthetic, indent=2), encoding="utf-8")
    np.savez_compressed(output/"ut_offset_code5_parity.npz", **arrays)


if __name__ == "__main__":
    main()
