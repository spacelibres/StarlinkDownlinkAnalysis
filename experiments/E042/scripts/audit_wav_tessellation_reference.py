"""Predict held WAV frequencies using the frozen E025 sequence and four gains."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from audit_wav_tessellation import CARRIERS, PERIOD, coherence, forward


def fit_and_score(matrix, symbols, calibration, held, test):
    channel = np.einsum("ncs,cs->nc", matrix, symbols)
    gain = np.linalg.lstsq(channel[~test], calibration[~test], rcond=1e-8)[0]
    prediction = channel[test] @ gain
    reference = held[test]
    residual = prediction - reference
    return {"coherence": float(coherence(prediction, reference)),
            "normalized_squared_error": float(np.vdot(residual, residual).real/np.vdot(reference, reference).real),
            "gain": [[float(a.real), float(a.imag)] for a in gain]}


def main():
    require_run_context()
    used = {}
    def artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, used)
    folded = artifact("v039-wav-joint-phase", "E041", "wav_joint_phase.npz")
    source = artifact("v040-wav-tessellation", "E042", "wav_tessellation.npz")
    r = read(artifact("v040-wav-tessellation", "E042", "wav_tessellation.json"))
    wide = read(artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json"))
    signed = np.asarray(wide["cycle_folds"][0]["signed_template"])
    reference = r["reference_comparison"]
    phase = reference["all_phases"][0]["phase_subtracted"]
    resource = np.asarray(reference["resource_index_by_lane_and_slot_class"])
    symbols = signed[(resource-phase) % len(signed)]
    with np.load(source, allow_pickle=False) as a:
        mask = a["held_harmonic_mask"]
        geometry = forward(a["harmonic_rows"], r["fitted_offset_hz"]).reshape(len(mask), len(CARRIERS), PERIOD)
    with np.load(folded, allow_pickle=False) as a:
        raw = a["mode_0_source_0_aligned_spectra"].reshape(len(a["centers_s"]), -1)
        cal, held = (raw[r["groups"][k]].mean(axis=0) for k in ("training", "held"))
    original = fit_and_score(geometry, symbols, cal, held, mask)
    alternates = []
    for row, col in np.ndindex(symbols.shape):
        other = symbols.copy()
        other[row, col] *= -1
        quality = fit_and_score(geometry, other, cal, held, mask)
        alternates.append({"carrier": int(CARRIERS[row]), "slot_class": col,
                           "resource_mod_60": int(resource[row, col]), **quality})
    result = {"schema": "starlink-wav-tessellation-reference/v1",
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in used.items()},
              "reference_phase_subtracted": phase, "reference": original, "single_bit_flip_models": alternates,
              "scope": "The reference is frozen after the receive-only fit. Each control fits four gains independently."}
    out = experiment_output("E042")
    out.mkdir(parents=True, exist_ok=True)
    (out/"wav_tessellation_reference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Reference four-gain held coherence", original["coherence"], flush=True)


if __name__ == "__main__":
    main()
