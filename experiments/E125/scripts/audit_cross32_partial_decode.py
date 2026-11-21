"""Measure partial cross-32QAM decoding against reserved sample-domain witnesses."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH, LIFT, checks
from audit_mcs66_partial_decoder import (
    CHECKPOINTS, GUARD_BITS, NORMALIZATION, RAW_VARIANCE, constellation_llr,
    min_sum, output_decisions, pn_metrics, syndrome_count,
)
from audit_mcs85_records import WEIGHT, detect_records, system_bytes
from audit_qam_pn15 import equality_predictions
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from extract_mcs66_padding_sweep import measure_tail
from receiver_models import pn_bits


FORMATS = {85: (66, 12288), 88: (70, 12800)}
RESOURCES = 3277


def parity_messages(incoming):
    """Exact even-parity extrinsic LLR with stable excluded products."""
    t = np.tanh(incoming / 2)
    product = np.ones_like(t)
    np.cumprod(t[..., :-1], axis=-1, out=product[..., 1:])
    product[..., :-1] *= np.cumprod(t[..., :0:-1], axis=-1)[..., ::-1]
    return 2 * np.arctanh(np.clip(product, -0.9999999, 0.9999999))


def sum_product(prior, support, iterations):
    edges = support.ravel()
    incidence = csr_matrix((np.ones(len(edges), dtype=prior.dtype), (edges, np.arange(len(edges)))),
                           shape=(prior.shape[1], len(edges)))
    posterior = prior.copy()
    for start in range(0, len(prior), 8):
        original = prior[start:start + 8]
        q = original[:, support]
        for _ in range(iterations):
            sent = parity_messages(q)
            full = original + (incidence @ sent.reshape(len(original), -1).T).T
            q = full[:, support] - sent
        posterior[start:start + len(original)] = full
    return posterior


def decode(prior, support, method, iterations):
    if iterations == 0:
        return prior
    if method == "min_sum":
        return min_sum(prior, support, NORMALIZATION, iterations)
    return sum_product(prior, support, iterations)


def reference_metrics(bits, seen, original_mask, pn, eligible, groups):
    if not np.any(eligible & original_mask):
        return None
    return pn_metrics(bits, seen, original_mask, pn, eligible, groups)


def settings(prior, bits, mask, groups, fit, hold, eligible, pn, check, shifted, inherited=None):
    def metric(posterior, subset):
        out, visible = output_decisions(posterior, bits[subset], mask[subset])
        return reference_metrics(out, visible, mask[subset], pn, eligible[subset], groups[subset])

    table = []
    for name, it in [("channel", 0)] + [(m, i) for m in ("min_sum", "sum_product") for i in CHECKPOINTS if i]:
        p = decode(prior, check, name, it)
        table.append({"method": name, "iterations": it, "calibration": metric(p[fit], fit),
                      "held": metric(p[hold], hold)})
    eligible_models = [row for row in table if row["calibration"] is not None]
    if eligible_models:
        selected = min(eligible_models, key=lambda r: (r["calibration"]["group_error_rate"], r["iterations"]))
        chosen = (selected["method"], selected["iterations"])
    else:
        chosen = inherited
    if chosen is None:
        raise ValueError("No calibration or transferred decoder preset")
    posterior = decode(prior, check, *chosen)
    wrong = decode(prior, shifted, *chosen)
    hard = reference_metrics(bits[hold], mask[hold], mask[hold], pn, eligible[hold], groups[hold])
    decoded, visible = output_decisions(posterior, bits, mask)
    wrong_bits, wrong_visible = output_decisions(wrong, bits, mask)
    observed_checks = mask[hold][:, check].all(axis=-1)
    syndrome = {k: int(syndrome_count(v, check, observed_checks).sum())
                for k, v in (("hard", bits[hold]), ("decoded", decoded[hold]))}
    return ({"table": table, "chosen": list(chosen), "chosen_used_inherited_setting": bool(not eligible_models),
             "hard": hard, "wrong_shifted_systematic": metric(wrong[hold], hold),
             "originally_observed_syndrome_rows": int(observed_checks.sum()), "violations": syndrome},
            {"decoded": (decoded, visible), "shifted": (wrong_bits, wrong_visible)})


def archive_inputs(code, gain, obs, data, pn, edge):
    key = f"{code}_terminal"
    rows = obs["archive"][key]["rows"]
    allowed = {(r["frame"], a["start"] + w * RESOURCES) for r in gain["frames"] for a in r["allocations"]
               if a["label"] == code for w in a["masked_transfer_stop_words"]}
    selected = np.array([(row["frame"], row["resource"]) in allowed for row in rows])
    rows = [row for row, keep in zip(rows, selected, strict=True) if keep]
    bits, seen = (np.unpackbits(data[f"archive_{key}_{k}"][selected], axis=1)[:, :BIT_LENGTH].astype(bool)
                  for k in ("bits", "seen"))
    bounds = [measure_tail(x, m, pn, edge) for x, m in zip(bits, seen, strict=True)]
    rows = [{**r, **b} for r, b in zip(rows, bounds, strict=True)]
    indices = np.arange(BIT_LENGTH)
    eligible = (indices >= np.array([b["bound"] for b in rows])[:, None] + GUARD_BITS) & (indices < edge)
    return rows, bits, seen, eligible


def lane_prior(bits, seen, fit, eligible, pn):
    members = np.broadcast_to(np.arange(BIT_LENGTH) % 5, bits.shape)
    known = fit[:, None] & seen & eligible
    count = np.bincount(members[known], minlength=5)
    wrong = np.bincount(members[known & (bits != pn)], minlength=5)
    if np.any(count == 0) or np.any(2 * wrong >= count):
        raise ValueError("Missing or invalid lane-calibration probabilities")
    rate = (wrong + .5) / (count + 1)
    strength = np.log((1 - rate) / rate).astype(np.float32)
    llr = np.where(bits, -strength[members], strength[members]) * seen
    return llr, {"observed": count.tolist(), "errors": wrong.tolist(), "llr": strength.tolist()}


def raw_inputs(code, evidence, labels, edge):
    points = np.array([complex(r["i"], r["q"]) for r in labels["cross32_mapping"]]) / np.sqrt(20)
    mapping = labels_as_bits(np.array([r["label"] for r in labels["cross32_mapping"]]), 5)
    if not np.allclose(points, evidence["cross32_points"]):
        raise ValueError("Raw capture constellation order changed")
    samples = evidence[f"mcs{code}_symbols"]
    frames = evidence[f"mcs{code}_frames"]
    llr = np.concatenate([constellation_llr(x, points, mapping, RAW_VARIANCE)
                          for x in np.array_split(samples, (len(samples) + 7) // 8)])
    bits = mapping[nearest_symbols(samples, points)].reshape(len(samples), -1)[:, :BIT_LENGTH].astype(bool)
    llr = llr.reshape(len(samples), -1)[:, :BIT_LENGTH].astype(np.float32)
    known = np.zeros_like(bits)
    for row in labels["word_results"]:
        if row["format"] != code or not row["accepted"]:
            continue
        i = row["candidate_index"]
        if row["frame"] != frames[i] or not row["allocation_final_word"]:
            raise ValueError("PN witness does not match frozen raw terminal word")
        known[i, row["start"] + GUARD_BITS:min(edge, row["stop"] - GUARD_BITS)] = True
    return frames, llr, bits, known


def raw_record_metrics(fit, frames, original, branches, pn, edge):
    masks = [np.frombuffer(bytes.fromhex(fit[k + "_hex"]), np.uint8)
             for k in ("template", "detection_mask", "verification_mask")]
    value, _, verify = masks
    result = []
    for frame in fit["frame_results"]:
        number = frame["frame"]
        if number in fit["frames_split"]["calibration"]:
            continue
        take = frames == number
        raw = system_bytes(original[take], edge, pn).ravel()
        coordinates = {r["byte_offset"]: r for r in frame["records"]}
        for pos, row in coordinates.items():
            if raw[pos:pos + len(value)].tobytes().hex() != row["observed_hex"]:
                raise ValueError("Raw record differs from the frozen hard-decision receipt")
        held = {"frame": number, "reference_records": len(coordinates), "observed_scores": {}}
        miss = {r["predicted"] for r in frame["length_predictions"] if r["status"] == "not_detected"}
        for name, output in {"nearest": original, **branches}.items():
            data = system_bytes(output[take], edge, pn).ravel()
            def verify_errors(pos):
                return int(WEIGHT[(data[pos:pos + len(value)] ^ value) & verify].sum())
            current = {r["byte_offset"] for r in detect_records(data, *masks, max_distance=fit["detection_max_errors"])}
            held["observed_scores"][name] = {
                "verification_errors_at_frozen_sites": sum(verify_errors(pos) for pos in coordinates),
                "frozen_sites_redetected": len(current & coordinates.keys()),
                "missed_successors_now_detected": [
                    {"position": pos, "verification_errors": verify_errors(pos),
                     "nearest_verification_errors": int(WEIGHT[(raw[pos:pos + len(value)] ^ value) & verify].sum()),
                     "observed_hex": data[pos:pos + len(value)].tobytes().hex()} for pos in sorted(current & miss)
                ],
                "new_scan_sites": sorted(current - coordinates.keys()),
            }
        result.append(held)
    return result


def main():
    require_run_context()
    inputs, packed = {}, {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    fit_record = read(artifact("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    if fit_record["pn_model"] != labels["frozen_pn_model"]:
        raise ValueError("PN model differs between raw record and modulation receipts")
    gain = read(artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    archive_meta = read(artifact("v120-cross-mcs-checks", "E122", "cross_mcs_checks.json"))
    earlier = read(artifact("v122-mcs85-archive-records-verified", "E124", "mcs85_archive_records.json"))
    excluded = earlier["archive_overlapping_groups_excluded"]
    matrices = checks(artifact)
    with np.load(artifact("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz"),
                 allow_pickle=False) as data:
        matrices[66] = data["row_support"]
    report = {"schema": "starlink-cross32-partial-decoder/v1", "pn_guard_bits": GUARD_BITS,
              "archive_split_seed": archive_meta["archive_split_seed"], "archive_excluded_groups": excluded,
              "min_sum_normalization": NORMALIZATION, "raw_variance": RAW_VARIANCE, "formats": {}}
    raw_source = artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    archive_source = artifact("v120-cross-mcs-checks", "E122", "cross_mcs_observations.npz")
    inherited = None
    with np.load(archive_source, allow_pickle=False) as archive, np.load(raw_source, allow_pickle=False) as raw:
        for code, (origin, edge) in FORMATS.items():
            check = matrices[origin].reshape(-1, 14)
            shifted = np.where(check < edge, check // LIFT * LIFT + (check + 1) % LIFT, check)
            rows, bits, seen, known = archive_inputs(code, gain, archive_meta, archive, pn, edge)
            groups = np.array([r["group"] for r in rows])
            included = ~np.isin(groups, excluded)
            training = np.array([r["calibration"] for r in rows])
            fit, hold = training & included, ~training & included
            prior, calibration = lane_prior(bits, seen, fit, known, pn)
            arch_report, _ = settings(prior, bits, seen, groups, fit, hold, known, pn, check, shifted)
            arch_report.update(rows=rows, lane_calibration=calibration)
            frames, prior, bits, known = raw_inputs(code, raw, labels, edge)
            fit = np.isin(frames, labels["frame_split"]["calibration"])
            raw_report, modes = settings(prior, bits, np.ones_like(bits), frames, fit, ~fit, known, pn, check, shifted,
                                         inherited=inherited)
            raw_report.update(frames=frames.tolist(), raw_group_assignment=labels["frame_split"])
            if code == 85:
                inherited = raw_report["chosen"]
                before = output_decisions(prior, bits, np.ones_like(bits))[0]
                raw_report["record_scores"] = raw_record_metrics(
                    fit_record, frames, bits, {"channel": before, **{k: v[0] for k, v in modes.items()}}, pn, edge
                )
            for name, pair in modes.items():
                for tag, values in zip(("bits", "seen"), pair, strict=True):
                    packed[f"{code}_{name}_{tag}"] = np.packbits(values, axis=1)
            report["formats"][str(code)] = {"archive": arch_report, "raw250": raw_report}
            print(f"MCS {code}: archive {arch_report['chosen']}, raw {raw_report['chosen']}", flush=True)
    report["inputs_sha256"] = {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}
    path = experiment_output("E125")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "cross32_partial_decisions.npz", **packed)
    (path / "cross32_partial_decode.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
