"""Measure Qin archive endpoints and validate new format widths and PN15 transfers."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from allocation_core import BIT_MODELS, UNITS
from audit_qam_pn15 import equality_predictions, matching_run
from audit_qin_exemplars import remove_header_gain
from audit_ut_binary_tail import BPSK, QAM16
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from validate_code37 import constellation_for

SEED = 2026112135
SNAPSHOT = "v032-qin-snapshot"
AUDIT = "v032-qin-exemplar-audit"
NEW_CODES = (6, 35, 42, 75, 81)
CROSS_CODES = (71, 75, 78, 81, 85, 88, 92)
FIT_TAIL = 600
MIN_COHERENCE = .65
MIN_RUN = 240
TERMINAL_MARGIN = 60


def terminal_edge(values, template, *, short=False):
    pos = np.arange(len(values))
    lag = np.arange(len(template))
    window = len(lag) if short else FIT_TAIL
    suffix = values[-window:]
    fit = (template[(pos[None, -window:] - lag[:, None]) % len(lag)] @ suffix) / len(suffix)
    phase = int(np.argmax(abs(fit)))
    coherence = float(abs(fit[phase]) / max(np.mean(abs(suffix)), 1e-30))
    result = {"phase_refined_edge": None, "fitted_phase": phase,
              "normalized_coherence": coherence, "run": None}
    if coherence < (.8 if short else MIN_COHERENCE):
        return result
    predicted_negative = template[(pos - phase) % len(lag)] < 0
    sampled_negative = (values * np.conj(fit[phase])).real < 0
    errors = sampled_negative != predicted_negative
    if short:
        reserved_period_errors = int(errors[-2*window:-window].sum())
        result["previous_period_errors"] = reserved_period_errors
        if reserved_period_errors > 3:
            return result
    peak = matching_run(errors)
    result["run"] = peak
    if (peak["stop"] >= len(pos) - (5 if short else TERMINAL_MARGIN)
            and peak["length"] >= (2*window if short else MIN_RUN)
            and (20 if short else 10) * peak["mismatches"] <= peak["length"]):
        start = peak["start"]
        half = len(lag) // 2
        refined = start + (phase - start + half) % len(lag) - half
        if 0 <= refined < len(pos):
            result["phase_refined_edge"] = refined
    return result


def detect_terminal(values, template):
    long = terminal_edge(values, template)
    if long["phase_refined_edge"] is not None:
        return {**long, "used_fit": "long"}
    return {**terminal_edge(values, template, short=True), "used_fit": "short", "long_attempt": long}


def header_geometry(row):
    header = row["h2"]
    if header.get("crc_ok") is not True or not header["layout20"]["packing_valid"]:
        raise ValueError("Archive header not validated or allocation layout fails")
    quantities = {x["mcs_candidate"]: x["quantity"] for x in header["layout20"]["entries"]}
    if len(quantities) != len(header["layout20"]["entries"]):
        raise ValueError("Duplicate descriptor in one allocation map")
    return row["h2_start"] + 114 * header["blocks"], quantities


def group_design(frames, headers):
    groups = np.array(sorted({x["group"] for x in frames}))
    locations = {c: {x["group"] for x in frames if c in header_geometry(headers[x["frame"]])[1]}
                 for c in NEW_CODES}
    required = set().union(*(s for s in locations.values() if len(s) == 1))
    order = np.random.default_rng(SEED).permutation(groups).tolist()
    calibration = required.copy()
    table = [(x["group"], [header_geometry(headers[x["frame"]])[1].get(c, 0) for c in NEW_CODES]) for x in frames]
    for g in [None, *order]:
        if g is not None:
            calibration.add(g)
        design = np.array([counts for group, counts in table if group in calibration])
        if len(design) and np.linalg.matrix_rank(design) == len(NEW_CODES):
            break
    reserved = set(groups.tolist()) - calibration
    return {"calibration_groups": sorted(calibration), "reserved_groups": sorted(reserved),
            "unique_group_codes": sorted(c for c, s in locations.items() if len(s) == 1),
            "seed": SEED, "randomized_order": order}


def fit_units(frame_metrics, headers, inherited, partition):
    design = []
    observed = []
    documented = []
    for sample in frame_metrics:
        edge = sample["nominal"]["phase_refined_edge"]
        if sample["group"] not in partition["calibration_groups"] or edge is None:
            continue
        start, counts = header_geometry(headers[sample["frame"]])
        background = start + sum(inherited.get(k, 0) * n for k, n in counts.items())
        design.append([counts.get(c, 0) for c in NEW_CODES])
        observed.append(edge - background)
        if any(design[-1]):
            documented.append(sample["frame"])
    design = np.array(design)
    if np.linalg.matrix_rank(design) != len(NEW_CODES):
        raise ValueError("Measured calibration endpoints do not identify the candidate widths")
    units = np.rint(np.linalg.lstsq(design, observed, rcond=None)[0]).astype(int)
    if min(units) <= 0:
        raise ValueError("Inferred nonpositive unit")
    errors = (design @ units - observed).tolist()
    return dict(zip(NEW_CODES, units.tolist(), strict=True)), documented, errors


def regions_for(row, widths):
    cursor, quantities = header_geometry(row)
    rows = []
    for code, count in quantities.items():
        n = count * widths[code]
        rows.append((code, count, cursor, cursor + n))
        cursor += n
    return rows


def joint_endpoint(row, lengths):
    cursor, quantities = header_geometry(row)
    models = {**BIT_MODELS, 35: (16384, 3), **{c: (16384, 5) for c in CROSS_CODES}}
    for c, n in quantities.items():
        if c in models:
            bits, width = models[c]
            cursor += (n * bits + width - 1) // width
        else:
            cursor += n * lengths[c]
    return cursor


def pdu_stream(snapshot, groups, p, signs):
    for group, path in zip(groups, snapshot, strict=True):
        with np.load(path, allow_pickle=False) as archive:
            symbols = archive["symbols"]
            for frame, data in zip(archive["frame"], symbols, strict=True):
                stored = data[1:, NONPILOT % 1024]
                yield int(frame), group, stored, remove_header_gain(stored * p, signs)


def alphabet_cost(values, bank):
    probe = values[::23]
    return {name: float(np.min(abs(probe[:, None] - points[None, :] / max(abs(points))) ** 2, axis=1).mean())
            for name, points in bank.items()}


def cross32_words(flat, start, stop, count, alphabet, mapping, pn, masks):
    width = 3277
    if stop - start != width * count:
        raise ValueError("Cross 32QAM resource dimension changed")
    rows = []
    received = []
    for j in range(count):
        point = start + j*width
        picked = {}
        for offset in (0, -1, 1):
            values = flat[point+offset:point+offset+width] * max(abs(alphabet))
            nearest = nearest_symbols(values[None, :], alphabet)[0]
            bits = labels_as_bits(mapping[nearest], 5).reshape(-1)[:16384]
            picked[offset] = bits
        hard = picked[0]
        received.append(hard)
        matched = matching_run(hard ^ pn)
        controls = {name: matching_run(hard ^ wrong) for name, wrong in masks.items()}
        controls.update({f"resource_offset_{offset}": matching_run(picked[offset] ^ pn) for offset in (-1, 1)})
        rows.append({"word": j, "resource_start": point, "allocation_final_word": j == count-1,
                     "pn15": matched, "controls": controls})
    return rows, received


def boundary_for_words(frames, received, pn):
    candidates = {}
    for code in (75, 81):
        fitting_stops = [word["pn15"]["stop"] for row in frames if row["partition"] == "calibration"
                         for region in row["new_regions"] if region["code"] == code
                         for word in region["cross32_word_matches"] if word["pn15"]["accepted"]]
        if not fitting_stops:
            raise ValueError(f"No calibration witness for Code {code}")
        candidates[code] = int(256 * np.rint(np.median(fitting_stops) / 256))
    for row in frames:
        for region in row["new_regions"]:
            code = region["code"]
            if code not in candidates:
                continue
            stop = candidates[code]
            for word in region["cross32_word_matches"]:
                bits = received[row["frame"], code, word["word"]]
                err = bits ^ pn
                word["at_candidate_boundary"] = {"before_errors": int(err[stop-256:stop].sum()),
                                                   "after_errors": int(err[stop:stop+256].sum())}
    return candidates


def main():
    require_run_context()
    inputs = {}
    source = read(checked_artifact(SNAPSHOT, "E034", "qin_snapshot_source.json", inputs))
    groups = source["hard_groups"]
    paths = [checked_artifact(SNAPSHOT, "E034", f"qin_hard_{g}.npz", inputs) for g in groups]
    audit = read(checked_artifact(AUDIT, "E034", "qin_exemplar_audit.json", inputs))
    headers = {x["frame"]: x for x in audit["headers"]}
    precedent = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    qam = read(checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs))
    template = np.array(precedent["cycle_folds"][0]["signed_template"], dtype=float)
    wrong_template = np.random.default_rng(SEED+1).permutation(template)
    inherited = {**UNITS, **{int(c): n for c, n in precedent["conditional_length_audit"]["candidate_resource_units"].items()}}
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    slots = np.arange(2, 302)
    p = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    wrong_p = np.column_stack([pn_reference(int(k), slots+1) for k in NONPILOT]).conj()
    observations = []
    for frame, group, stored, flat in pdu_stream(paths, groups, p, signs):
        controls = {"wrong_slot_pn": detect_terminal(remove_header_gain(stored * wrong_p, signs), template),
                    "permuted_template": detect_terminal(flat, wrong_template)}
        observations.append({"frame": frame, "group": group, "nominal": detect_terminal(flat, template),
                             "controls": controls})
    partition = group_design(observations, headers)
    learned, fit_frames, fit_errors = fit_units(observations, headers, inherited, partition)
    lengths = {**inherited, **learned}
    points32 = np.array([complex(x["i"], x["q"]) for x in qam["cross32_mapping"]]) / np.sqrt(20)
    mapping = np.array([x["label"] for x in qam["cross32_mapping"]])
    bank = {"BPSK": BPSK, "QPSK": np.exp(1j*(np.arange(4)+.5)*np.pi/2),
            "eight_point": constellation_for(2, False, True), "square16": QAM16, "cross32": points32}
    parameters = qam["frozen_pn_model"]
    coords = np.arange(16384)
    pn = equality_predictions(coords, pn_bits(), parameters)
    wrong = {f"pn_phase_{step}": equality_predictions(coords, pn_bits(), {**parameters, "phase": parameters["phase"]+step})
             for step in (-1, 1)}
    by_frame = {x["frame"]: x for x in observations}
    hard_words = {}
    for frame, group, _, flat in pdu_stream(paths, groups, p, signs):
        row = by_frame[frame]
        edges = regions_for(headers[frame], lengths)
        predicted = edges[-1][-1]
        observed = row["nominal"]["phase_refined_edge"]
        joint = joint_endpoint(headers[frame], lengths)
        row.update(partition="calibration" if group in partition["calibration_groups"] else "reserved",
                   per_word_end=predicted, joint_rounding_end=joint, detected_minus_prediction=(
                       observed - predicted if observed is not None else None), new_regions=[])
        for code, count, start, stop in edges:
            if code not in NEW_CODES:
                continue
            metrics = {"code": code, "words": count, "start": start, "stop": stop,
                       "mean_nearest_point_cost": alphabet_cost(flat[start:stop], bank)}
            if code in (75, 81):
                rows, received = cross32_words(flat, start, stop, count, points32, mapping, pn, wrong)
                metrics["cross32_word_matches"] = rows
                hard_words.update({(frame, code, j): bits for j, bits in enumerate(received)})
            row["new_regions"].append(metrics)
    stops = boundary_for_words(observations, hard_words, pn)
    out = experiment_output("E035")
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "starlink-qin-lengths/v1",
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "partition": partition, "inherited_units": inherited, "fitted_units": learned,
              "calibration_frames_with_new_formats": fit_frames, "calibration_residuals": fit_errors,
              "terminal_fit_symbols": FIT_TAIL, "terminal_coherence_min": MIN_COHERENCE,
              "terminal_min_length": MIN_RUN, "permutation_seed": SEED+1, "frames": observations,
              "cross32_systematic_boundary_candidates": stops, "boundary_window_bits": 256,
              "scope": "Capture-specific whole-group validation of resource lengths. Nominal bits and PN matching do not verify parity matrices."}
    (out / "qin_lengths.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    measured = [x for x in observations if x["detected_minus_prediction"] is not None]
    print("Resource units:", learned, "; matched detected endpoints",
          sum(x["detected_minus_prediction"] == 0 for x in measured), "of", len(measured), flush=True)


if __name__ == "__main__":
    main()
