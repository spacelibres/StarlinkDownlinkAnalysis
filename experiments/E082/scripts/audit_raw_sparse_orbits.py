"""Transfer frozen MCS 66 parity projections to the original wideband IQ replay."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_gain_calibrated_folds import EXTRACTION
from audit_qam_pn15 import collect, equality_predictions, matching_run
from audit_sparse_qc_orbits import orbit_positions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


def assess(values, positions, constants):
    parity = np.bitwise_xor.reduce(values[positions], axis=-1)
    seen = np.array([x is not None for x in constants])
    sign = np.array([bool(x) for x in constants])
    matched = (parity == sign) & seen
    contrast = seen & np.roll(seen, 1)
    return {"matched": int(matched.sum()), "compared": int(seen.sum()),
            "shifted_one_matched": int(((parity == np.roll(sign, 1)) & contrast).sum()),
            "shifted_one_compared": int(contrast.sum())}


def main():
    require_run_context()
    inputs = {}

    def load(run, experiment, name):
        return read(checked_artifact(run, experiment, name, inputs))

    qam_run = "v026-ut-qam-extract-repeat"
    source = checked_artifact(qam_run, "E028", "ut_qam_symbols.npz", inputs)
    receiver = load(qam_run, "E028", "ut_qam_extraction.json")
    data, _, _ = collect(source, receiver)
    refined, original, _, witnesses = data[66]
    orbit_run = "v080-sparse-qc-orbits"
    orbits = load(orbit_run, "E082", "sparse_qc_orbits.json")["outcomes"]["66"]
    admitted = [row for row in orbits if row["length"] == 512]
    qualified = load(EXTRACTION, "E082", "gain_padded_words.json")["cohorts"]["short_66"]
    fit = {(r["frame"], r["resource"]) for r in qualified if r["calibration"]}
    included = {(r["frame"], r["resource"]) for r in qualified}
    cross = load("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json")
    reference = equality_predictions(np.arange(refined.shape[1]), pn_bits(), cross["frozen_pn_model"])
    overlap_record = load("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json")["overlap"]
    overlap = {row["ut_candidate"]: row["qin_frame"] for row in overlap_record["pairs"]}
    edge = 12288
    hypothesis = [(arm, edge + orbit_positions(arm["shape"], 512)) for arm in admitted]
    rows = []
    for bits, base, witness in zip(refined, original, witnesses, strict=True):
        qin_index = overlap.get(witness["frame"])
        position = (qin_index, witness["resource_start"])
        row = {**witness, "archive_overlap_frame": qin_index, "matched_word_in_orbit_fit": position in fit,
               "matched_word_in_short_pn_cohort": position in included, "pn_interval": matching_run(bits != reference),
               "arms": [{"shape": arm["shape"], "refined": assess(bits, positions, arm["fixed_affine_constants"]),
                         "baseline": assess(base, positions, arm["fixed_affine_constants"])}
                        for arm, positions in hypothesis]}
        rows.append(row)
    report = {"schema": "starlink-raw-sparse-orbits/v1", "mcs": 66, "rows": rows,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E082")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_sparse_orbits.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in rows:
        if row["last_word_of_allocation"]:
            print(row["frame"], row["archive_overlap_frame"], row["pn_interval"]["start"],
                  [arm["refined"] for arm in row["arms"]])


if __name__ == "__main__":
    main()
