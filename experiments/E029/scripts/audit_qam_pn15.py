"""Discover and test a byte-ordered PN15 model for repeated QAM coordinates."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from audit_shortening_gaps import SOURCES as LEGACY_GAPS
from audit_ut_qam_fec import slice_labels
from gap_hypotheses_core import phase_fit, trajectory
from receiver_models import pn_bits

SOURCE_RUN = ROOT / "runs/v026-ut-qam-extract-repeat"
DATA = SOURCE_RUN / "artifacts/experiments/E028/results"
WIDTH = 16384
MIN_LENGTH = 512


def ordered_indices(position, width):
    """Reverse positions within a width-bit unit, retaining unit order."""
    return position // width * width + width - 1 - position % width


def matching_run(differences):
    """Maximum-scoring contiguous run under fixed 0.97 versus 0.5 match models."""
    scores = np.where(differences, np.log(.03 / .5), np.log(.97 / .5))
    prefix = np.r_[0., np.cumsum(scores)]
    score = prefix - np.minimum.accumulate(prefix)
    stop = int(np.argmax(score))
    start = int(np.argmin(prefix[:stop + 1]))
    errors = int(np.count_nonzero(differences[start:stop]))
    length = stop - start
    return {
        "start": start, "stop": stop, "length": length, "mismatches": errors,
        "log_score_gain": float(score[stop]), "accepted": length >= MIN_LENGTH and errors <= .1 * length,
    }


def collect(path, receiver):
    frames = set(receiver["calibration_frames"])
    datasets = {}
    with np.load(path, allow_pickle=False) as archive:
        points, mapping = archive["constellation_points"], archive["frozen_gray_mapping"]
        for code in (66, 70):
            group = archive[f"mcs{code}_frames"]
            obs = archive[f"mcs{code}_refined"][:, 1:-1]
            baseline = archive[f"mcs{code}_baseline"][:, 1:-1]
            bits = slice_labels(obs, points, mapping)[0].reshape(len(obs), WIDTH)
            initial = slice_labels(baseline, points, mapping)[0].reshape(len(obs), WIDTH)
            witnesses = []
            for i, (f, start) in enumerate(zip(group, archive[f"mcs{code}_starts"], strict=True)):
                last = bool(i == len(group) - 1 or f != group[i + 1])
                witnesses.append({"format": code, "candidate_index": i, "frame": int(f),
                                  "resource_start": int(start), "last_word_of_allocation": last,
                                  "calibration": int(f) in frames})
            datasets[code] = (bits, initial, obs, witnesses)
    return datasets, points, mapping


def repeat_search(datasets):
    info = {}
    for l, r in ((66, 66), (70, 70), (66, 70)):
        xl, _, _, a = datasets[l]
        xr, _, _, b = datasets[r]
        left = 1 - 2 * xl.astype(np.float32)
        right = 1 - 2 * xr.astype(np.float32)
        similarity = left @ right.T / WIDTH
        valid = np.array([x["frame"] for x in a])[:, None] != np.array([x["frame"] for x in b])[None, :]
        if l == r:
            valid &= np.triu(np.ones(similarity.shape, dtype=bool), 1)
        active = np.flatnonzero(valid.ravel())
        rank = active[np.argsort(-abs(similarity.ravel()[active]), kind="stable")]
        strong = int(np.count_nonzero(abs(similarity.ravel()[active]) > .15))
        pairs = []
        for idx in rank[:12]:
            i, j = np.unravel_index(idx, similarity.shape)
            pairs.append({"left": a[i], "right": b[j], "full_word_sign_correlation": float(similarity[i, j]),
                          "longest_match_model": matching_run(xl[i] != xr[j])})
        info[f"{l}_{r}"] = {"different_frame_pairs": len(active), "correlation_magnitude_over_point15": strong,
                            "highest_magnitude_pairs": pairs}
    return info


def fit_from_calibration(datasets, pn):
    left, _, _, meta66 = datasets[66]
    right, _, _, meta70 = datasets[70]
    a = [x for x in meta66 if x["calibration"] and x["last_word_of_allocation"]]
    b = [x for x in meta70 if x["calibration"] and x["last_word_of_allocation"]]
    pairs = []
    for x in a:
        for y in b:
            if x["frame"] == y["frame"]:
                continue
            run = matching_run(left[x["candidate_index"]] != right[y["candidate_index"]])
            if run["accepted"]:
                pairs.append({"left": x, "right": y, **run})
    source = max(pairs, key=lambda r: r["log_score_gain"])
    ia, ib = source["left"]["candidate_index"], source["right"]["candidate_index"]
    start, stop = source["start"], source["stop"]
    candidates = np.arange(start, stop)
    consensus = candidates[left[ia, candidates] == right[ib, candidates]]
    variants = []
    for width in (1, 2, 4, 8, 16, 32):
        trial = phase_fit(pn, ordered_indices(consensus, width), left[ia, consensus])
        variants.append({"within_unit_reverse": width, **trial})
    winner = max(variants, key=lambda v: v["training_hits"])
    report = {"calibration_pairs_with_long_runs": pairs, "fitting_pair": source,
              "consensus_coordinates": len(consensus), "pn_fit_candidates": variants, "winner": winner}
    return report


def equality_predictions(index, pn, fit):
    order = ordered_indices(index, fit["within_unit_reverse"])
    params = [fit[k] for k in ("phase", "direction", "complement")]
    return trajectory(pn, order, *params)


def audit_words(datasets, pn, fit, points, mapping):
    position = np.arange(WIDTH)
    model = equality_predictions(position, pn, fit)
    wrong = [equality_predictions(position, pn, {**fit, "phase": fit["phase"] + delta})
             for delta in (-1, 1)]
    # Exact symbol positions under the existing physical Gray convention.
    packed = mapping @ np.array([8, 4, 2, 1], dtype=np.uint8)
    labels = model.reshape(-1, 4) @ np.array([8, 4, 2, 1], dtype=np.uint8)
    wave = points[np.argsort(packed)[labels]]
    summary = {}
    records = []
    for format_, (bits, initial, symbols, metadata) in datasets.items():
        for x, y, z, meta in zip(bits, initial, symbols, metadata, strict=True):
            interval = matching_run(x != model)
            base = matching_run(y != model)
            controls = [matching_run(x != v) for v in wrong]
            row = {**meta, "match": interval, "baseline_match": base,
                   "wrong_phase_max_length": max(v["length"] for v in controls),
                   "wrong_phase_any_accepted": any(v["accepted"] for v in controls)}
            if interval["accepted"]:
                lo, hi = (interval["start"] + 3) // 4, interval["stop"] // 4
                row["matched_symbol_mse"] = float(np.mean(abs(z[lo:hi] - wave[lo:hi]) ** 2))
                row["next_symbol_mse"] = float(np.mean(abs(z[hi:hi + 128] - wave[hi:hi + 128]) ** 2))
            records.append(row)
        picked = [r for r in records if r["format"] == format_ and r["match"]["accepted"]]
        training = [r for r in picked if r["calibration"]]
        candidate_end = [int(round(x["match"]["stop"] / 256) * 256) for x in training]
        proposed = candidate_end[0] if candidate_end and len(set(candidate_end)) == 1 else None
        reserved = [r for r in picked if not r["calibration"]]
        summary[str(format_)] = {
            "accepted_calibration_frames": [r["frame"] for r in training],
            "accepted_reserved_frames": [r["frame"] for r in reserved],
            "reserved_matched_bits": sum(r["match"]["length"] for r in reserved),
            "reserved_bit_disagreements": sum(r["match"]["mismatches"] for r in reserved),
            "nonfinal_words_accepted": sum(not r["last_word_of_allocation"] for r in picked),
            "candidate_coarse_stop_in_calibration": proposed,
            "reserved_stop_errors_relative_to_candidate": [r["match"]["stop"] - proposed for r in reserved]
            if proposed is not None else [],
        }
    return records, summary, model


def legacy_transfer(pn, fit, hashes):
    """Test the new bit numbering without changing any preserved gap data."""
    results = []
    for code, (stem, expected, span, raw_key, final_key, _) in LEGACY_GAPS.items():
        path = ROOT / (stem + ".npz")
        if sha256(path) != expected:
            raise ValueError("Frozen legacy gap changed")
        hashes[path.relative_to(ROOT).as_posix()] = expected
        first, stop = span
        p = np.arange(first, stop)
        with np.load(path, allow_pickle=False) as archive:
            words = archive[raw_key][:, first:stop]
            if not np.array_equal(words, archive[final_key][:, first:stop]):
                raise ValueError("Legacy gap decisions changed")
            training = archive["random_training" if code == 28 else "training_frames"]
        fixed = equality_predictions(p, pn, fit)
        q = np.broadcast_to(ordered_indices(p, fit["within_unit_reverse"]), words.shape)
        frozen_training = phase_fit(pn, q[training], words[training])
        training_fit = {**fit, **frozen_training}
        reserved = equality_predictions(p, pn, training_fit)
        results.append({
            "code": code, "relative_bit_interval": list(span),
            "byte_order_and_phase_frozen_hits": int(np.count_nonzero(words == fixed)),
            "byte_order_and_phase_frozen_total": words.size,
            "legacy_calibrated_phase_model": frozen_training,
            "legacy_calibrated_reserved_hits": int(np.count_nonzero(words[~training] == reserved)),
            "legacy_calibrated_reserved_total": words[~training].size,
        })
    return results


def main():
    require_run_context()
    source = DATA / "ut_qam_symbols.npz"
    metadata = DATA / "ut_qam_extraction.json"
    receipt = SOURCE_RUN / "manifest.json"
    run_data = json.loads(receipt.read_text(encoding="utf-8"))
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (source, metadata, receipt)}
    for p in (source, metadata):
        if hashes[p.relative_to(ROOT).as_posix()] != run_data["changed_artifacts"][f"experiments/E028/results/{p.name}"]["sha256"]:
            raise ValueError("Input candidate observations changed")
    receiver = json.loads(metadata.read_text(encoding="utf-8"))
    datasets, points, mapping = collect(source, receiver)
    pn = pn_bits()
    fit = fit_from_calibration(datasets, pn)
    records, summary, model = audit_words(datasets, pn, fit["winner"], points, mapping)
    legacy = legacy_transfer(pn, fit["winner"], hashes)
    report = {
        "schema": "starlink-qam-bytewise-pn15/v1", "inputs_sha256": hashes,
        "pn_seed": "000000100000010", "pn_period": len(pn),
        "fit": fit, "per_format": summary, "words": records, "repetition_discovery": repeat_search(datasets),
        "legacy_shortening_gap_transfer": legacy,
        "candidate_pn_model_sha256": hashlib.sha256(np.packbits(model).tobytes()).hexdigest(),
        "scope": "Retrospective grouped test of a deterministic bit-domain pattern in receiver candidate words. "
                 "This does not by itself identify the layer, code matrix or application fields.",
    }
    out = experiment_output("E029")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qam_pn15_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(fit["winner"], summary, flush=True)


if __name__ == "__main__":
    main()
