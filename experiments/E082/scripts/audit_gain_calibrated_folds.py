"""Rerun frozen QC projections on the expanded, gain-corrected archive cohort."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_dense_padding_folds import LIFTS, RESIDUES, SEED, audit_model, family_controls, fold_word, npz_bits
from audit_ut_repeated_data import checked_artifact, read


EXTRACTION = "v080-gain-padded-words"


def folded_formats(meta, archive, groups):
    formats = {}
    for key, records in meta["cohorts"].items():
        if not key.startswith("short_"):
            continue
        label = key.split("_")[1]
        length = records[0]["length"]
        edge = records[0]["edge"]
        if length % 128 not in (0, 1) or any((r["length"], r["edge"]) != (length, edge) for r in records):
            raise ValueError(f"Inconsistent word geometry in {key}")
        end = length // 128 * 128
        width = meta["prefix_bits"]
        ids = np.array([r["group"] for r in records])
        bits = npz_bits(archive, key + "_bits", length)
        seen = npz_bits(archive, key + "_seen", length).astype(bool)
        output, visible = bits[:, edge:end], seen[:, edge:end]
        arrays = {"prefix": (bits[:, :width], seen[:, :width]),
                  "conditional_prefix": (npz_bits(archive, key + "_conditioned", width),
                                         npz_bits(archive, key + "_conditional_mask", width).astype(bool))}
        arms = {}
        for name, (prefix, prefix_seen) in arrays.items():
            arms[name] = [audit_model(prefix, prefix_seen, output, visible, ids, groups,
                                      lift, residue, np.random.default_rng(SEED))
                          for lift in LIFTS if not (edge % lift or (end - edge) % lift)
                          for residue in RESIDUES]
        formats[label] = {"words": len(records), "groups": len(np.unique(ids)), "arms": arms}
        print("Format", label, "words", len(records), "groups", len(np.unique(ids)), flush=True)
    return formats


def pn_sensitivity(meta, archive, pn):
    result = {}
    for key, rows in meta["cohorts"].items():
        if not key.startswith("short_"):
            continue
        edge, width = rows[0]["edge"], rows[0]["length"]
        interval = slice(edge - 512, edge)
        errors = npz_bits(archive, key + "_bits", width)[:, interval] ^ pn[interval]
        seen = npz_bits(archive, key + "_seen", width)[:, interval].astype(bool)
        held = np.array([not row["calibration"] for row in rows])
        outcomes = []
        for lift in (16, 32, 64, 128, 256, 512):
            for residue in RESIDUES:
                wrong, visible = fold_word(errors[held], seen[held], lift, residue)
                outcomes.append({"bits_per_block": lift, "residues": residue, "positions": visible.size,
                                 "observed": int(visible.sum()), "errors": int((wrong & visible).sum())})
        result[key] = outcomes
    return result


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    from audit_qam_pn15 import equality_predictions
    from receiver_models import pn_bits

    meta = load(EXTRACTION, "E082", "gain_padded_words.json")
    previous = load("v075-dense-padding-diagnostic", "E077", "dense_padding_folds.json")
    groups = previous["calibration_groups"]
    cross = load("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json")
    pn = equality_predictions(np.arange(16385), pn_bits(), cross["frozen_pn_model"])
    data = checked_artifact(EXTRACTION, "E082", "gain_padded_words.npz", inputs)
    with np.load(data, allow_pickle=False) as archive:
        formats = folded_formats(meta, archive, groups)
        sensitivity = pn_sensitivity(meta, archive, pn)
    reference_run = "v022-ut-replay"
    reference_meta = load(reference_run, "E014", "ut_transfer.json")["words"]
    data5 = checked_artifact(reference_run, "E014", "ut_observations.npz", inputs)
    with np.load(data5, allow_pickle=False) as z:
        bits = z["hard_code5_words"]
    frames = np.array([r["candidate_index"] for r in reference_meta])
    held5 = previous["code5_positive_control"]["training_frames"]
    masks = np.ones(bits[:, :1024].shape, dtype=bool)
    control = [audit_model(bits[:, :1024], masks, bits[:, 1024:], masks, frames, held5, 128, residue,
                           np.random.default_rng(SEED)) for residue in RESIDUES]
    families = {field: family_controls(formats, field) for field in ("wrong_group_controls", "wrong_position_controls")}
    result = {"schema": "starlink-gain-calibrated-folds/v1", "formats": formats, "code5_control": control,
              "family_controls": families, "known_pn_sensitivity_held": sensitivity, "seed": SEED,
              "calibration_groups": groups,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E082")
    out.mkdir(parents=True, exist_ok=True)
    (out / "gain_calibrated_folds.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print({key: [row["scores_equal_or_better_including_observed"], row["leading"]] for key, row in families.items()})


if __name__ == "__main__":
    main()
