"""Test frozen labels, PN15, and a finite bank of legacy encoder transfers."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code28_reencoding import fft_binary_parities
from audit_qam_pn15 import equality_predictions, matching_run
from audit_ut_binary_tail import sha256
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from code21_ring import binary_blocks, evaluate, pack_blocks
from code28_core import coordinates
from code28_receiver_core import calibration_consensus
from qc_figure_core import qc_syndrome
from receiver_models import pn_bits
from validate_code37 import constellation_for
from validate_qc_figure import constellation

SEED = 2026112136
TARGET = (6, 35, 42, 93)
N = 16384


def split_groups(groups, format_id, fit_93):
    if format_id == 93:
        fitting = np.array([fit_93])
    else:
        order = np.random.default_rng(SEED + format_id).permutation(np.unique(groups))
        fitting = order[:len(order)//2]
    training = np.isin(groups, fitting)
    return training, {"calibration": sorted(set(groups[training].tolist())),
                      "reserved": sorted(set(groups[~training].tolist()))}


def qam_words(symbols, code, constellation32, map32):
    if code == 6:
        return (symbols.real < 0).astype(np.uint8)
    if code == 35:
        points = constellation_for(2, False, True)
        mapping = labels_as_bits(np.arange(8), 3)
    elif code == 42:
        points, mapping = constellation(39)
    else:
        points, mapping = constellation32, map32
    ids = nearest_symbols(symbols * max(abs(points)), points)
    return mapping[ids].reshape(len(symbols), -1)


def check_pn(bits, pn, alternates, training):
    assessed = [matching_run(x ^ pn) for x in bits[:, :len(pn)]]
    peaks = [result["stop"] for result, train in zip(assessed, training, strict=True) if train and result["accepted"]]
    if not peaks:
        raise ValueError("No PN interval in calibration groups")
    stop = int(256 * np.rint(np.median(peaks) / 256))
    rows = []
    for w, result in zip(bits[:, :len(pn)], assessed, strict=True):
        errors = w ^ pn
        rows.append({"fixed": result, "control": {name: matching_run(w ^ seq) for name, seq in alternates.items()},
                     "before_errors": int(errors[stop-256:stop].sum()),
                     "after_errors": int(errors[stop:stop+256].sum())})
    return stop, peaks, rows


def lifted_code6_checks(bits, p9, p21, shifts):
    variants = {"whole_code10_sparse_syndrome": qc_syndrome(bits, shifts, 1024).sum(axis=1).tolist()}
    for name, ref in (("from_code9", p9), ("from_code21", p21)):
        for stride in (1, 1024 // ref.shape[-1]):
            trial = np.zeros((10, 6, 1024), dtype=np.uint8)
            trial[:, :, np.arange(ref.shape[-1])*stride] = ref
            p, _ = fft_binary_parities(bits[:, :10240].reshape(-1, 10, 1024), trial)
            variants[f"{name}_exponent_stride_{stride}"] = (p != bits[:, 10240:]).sum(axis=1).tolist()
    return variants


def code6_consensus(words, pn, rows, training):
    if np.count_nonzero(training) < 10:
        return {"status": "insufficient_training_words"}
    report = {}
    assisted = words.copy()
    for index, row in enumerate(rows):
        r = row["fixed"]
        if r["accepted"]:
            left, right = r["start"], min(10240, r["stop"])
            assisted[index, left:right] = pn[left:right]
    for label, bits in (("observed", words), ("reference_assisted", assisted)):
        p, trial = calibration_consensus(bits[training, :10240], bits[training, 10240:], 1024, SEED, trials=256)
        if p is not None:
            pred = binary_blocks(evaluate(pack_blocks(bits[~training, :10240], 1024), p, 1024), 1024)
            trial["held_parity_mismatches"] = (pred != bits[~training, 10240:]).sum(axis=1).tolist()
            trial["fitted_polynomials_hex"] = [[hex(value) for value in row] for row in p]
        report[label] = trial
    return report


def main():
    require_run_context()
    inputs = {}
    extract_run = "v034-qin-low-mcs-extract"
    extracted = read(checked_artifact(extract_run, "E036", "qin_low_mcs_extract.json", inputs))
    samples = checked_artifact(extract_run, "E036", "qin_low_mcs_symbols.npz", inputs)
    mapping = read(checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs))
    code9 = checked_artifact("v013-code9-time-unwrapped", "E015", "code9_observations.npz", inputs)
    code21 = checked_artifact("v014-code21-repeat", "E016", "code21_observations.npz", inputs)
    code28 = checked_artifact("v019-code28-reproduction-repeat", "E021", "code28_reproduction.npz", inputs)
    shifts_path = ROOT / "experiments/E018/figure12_shifts.json"
    inputs[shifts_path] = sha256(shifts_path)
    shifts = np.array(read(shifts_path)["1024"])
    freeze = mapping["frozen_pn_model"]
    sequence = pn_bits()
    indices = np.arange(N)
    p = equality_predictions(indices, sequence, freeze)
    alternates = {f"phase_{d}": equality_predictions(indices, sequence, {**freeze, "phase": freeze["phase"]+d})
                  for d in (-1, 1)}
    points32 = np.array([complex(x["i"], x["q"]) for x in mapping["cross32_mapping"]]) / np.sqrt(20)
    map32 = labels_as_bits(np.array([x["label"] for x in mapping["cross32_mapping"]]), 5)
    with np.load(code9, allow_pickle=False) as a, np.load(code21, allow_pickle=False) as b, np.load(code28, allow_pickle=False) as c:
        bank9 = a["parity_generator"][::128].reshape(10, 6, 128)
        bank21 = b["parity_generator"][::512].reshape(10, 6, 512)
        bank28 = c["generator_seed_bits"].reshape(9, 7, 1024)
    report = {}
    with np.load(samples, allow_pickle=False) as data:
        for c in TARGET:
            symbols = data[f"mcs{c}_symbols"]
            ids, groups = data[f"mcs{c}_frames"], data[f"mcs{c}_groups"]
            training, split = split_groups(groups, c, extracted["mcs93_fit_group"])
            bits = qam_words(symbols, c, points32, map32)
            stop, peaks, rows = check_pn(bits, p, alternates, training)
            frame_info = zip(ids, groups, data[f"mcs{c}_is_final"], training, strict=True)
            for row, (frame, group, last, use) in zip(rows, frame_info, strict=True):
                row.update(frame=int(frame), group=int(group), last=bool(last), calibration=bool(use))
            model_checks = {}
            if c == 6:
                model_checks = lifted_code6_checks(bits, bank9, bank21, shifts)
                model_checks["blind_qc_calibration"] = code6_consensus(bits, p, rows, training)
            if c == 35:
                u, v, _ = coordinates(bits)
                parity, _ = fft_binary_parities(u.reshape(-1, 9, 1024), bank28)
                model_checks["frozen_shortened_code28"] = (parity != v).sum(axis=1).tolist()
            if c == 42:
                model_checks["whole_code39_sparse_syndrome"] = qc_syndrome(bits, shifts, 1024).sum(axis=1).tolist()
            report[str(c)] = {"groups": split, "candidate_boundary": stop, "calibration_pn_stops": peaks,
                              "words": rows, "finite_legacy_models": model_checks}
            print(c, "words", len(rows), "PN hits", sum(row["fixed"]["accepted"] for row in rows),
                  "candidate", stop, flush=True)
    out = experiment_output("E036")
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema": "starlink-qin-low-mcs-audit/v1", "whole_group_seed": SEED,
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "pn_model": freeze, "boundary_comparison_bits_each_side": 256, "formats": report,
              "scope": "Frozen receiver-relative labels and finite encoder transfer. Code 93 uses the same calibration group as its width fit."}
    (out / "qin_low_mcs_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
