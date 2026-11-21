"""Test fractional frame placement on matched raw pilot and data samples."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_leo_pairs_live import REPORT
from audit_leo_pilot_phase_transfer import BLOCK_SLOTS, FIRST_SLOT
from audit_leo_tcode_iq_prediction import score_families, visible_carriers, waveform_bank
from audit_narrowband_cycle import permute_lanes
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_reference


SEED = 2026112190
MODELS = ("fractional", "rounded", "opposite_rounding")
CONTROL_WORDS = 9
GUARDS = (.6, 1.5)


def common_aperture(length, origins, rate, slot_s, cp_s, guard):
    first = max(0, int(np.ceil(min(origins) + FIRST_SLOT * slot_s * rate)))
    stop = min(length, int(np.ceil(max(origins) + 302 * slot_s * rate)))
    index = np.arange(first, stop)
    time = (index[None, :] - origins[:, None]) / rate
    slot = np.floor(time / slot_s + 1e-10).astype(int)
    local = time - slot * slot_s
    same_slot = (slot == slot[:1]).all(axis=0)
    guarded = ((local >= cp_s + guard / rate) & (local <= slot_s - guard / rate)).all(axis=0)
    valid = same_slot & guarded & (slot[0] >= FIRST_SLOT) & (slot[0] <= 301)
    return index[valid], slot[0, valid], local[:, valid]


def tone_atoms(local, frequencies, cp_s):
    return np.exp(2j * np.pi * (local[..., None] - cp_s) * frequencies) / np.sqrt(8)


def fit_prediction(design, samples, cal):
    return design @ np.linalg.lstsq(design[cal], samples[cal], rcond=1e-9)[0]


def error_energy(samples, prediction, cal):
    residual = (samples - prediction)[~cal]
    return float(np.vdot(residual, residual).real)


def frame_features(iq, rate, origins, frequencies, pilots, tones, cp, slot_s, cfo, delta, guard, train):
    index, slot, local = common_aperture(len(iq), origins, rate, slot_s, cp, guard)
    block = (slot - FIRST_SLOT) // BLOCK_SLOTS
    cal = np.isin(block, train)
    if not cal.any() or cal.all():
        raise ValueError("Both slot partitions must have common samples")
    t = index / rate
    samples = iq[index] * np.exp(-2j * np.pi * (cfo * t + delta * (t - t.mean())))
    carriers, outside = tones
    core = tone_atoms(local, frequencies, cp)
    pilot = core * pilots[slot - 2]
    earlier = core * pilots[slot - 3]
    later = core * np.vstack((pilots, np.zeros((1, 8))))[slot - 1]
    qam_pn = np.column_stack([pn_reference(int(k), slot) for k in carriers])
    atoms = tone_atoms(local, outside, cp) * qam_pn
    classes = (slot - 2) % 15
    expanded = atoms[..., None] * (classes[:, None, None] == np.arange(15))
    dc = np.ones(len(index))
    gains = [np.linalg.lstsq(np.column_stack((mat[cal], dc[cal])), samples[cal], rcond=1e-9)[0]
             for mat in pilot]
    return {
        "cal": cal, "samples": samples, "dc": dc, "pilot": pilot, "basis_gains": np.asarray(gains),
        "nuisance": np.concatenate((atoms, earlier, later), axis=-1), "expanded": expanded,
        "slots": np.unique(slot).tolist(),
    }


def score_frame(features, channels, waves):
    cal, data, dc = (features[k] for k in ("cal", "samples", "dc"))
    rows = {}
    for arm, name in enumerate(MODELS):
        basis = features["pilot"][arm]
        gains = features["basis_gains"][arm]
        pilot = basis @ gains[:-1] + gains[-1]
        shared = fit_prediction(np.column_stack((basis @ channels[arm], dc)), data, cal)
        background = features["nuisance"][arm]
        ref = features["expanded"][arm]
        # All fits use the calibration slots from this frame. Channel shape was
        # frozen on a disjoint group of complete frames.
        available = {
            "free_pilot_channel": score_families(np.column_stack((basis, dc, background)), ref, data, cal, waves),
            "shared_pilot_channel": score_families(background, ref, data - shared, cal, waves),
        }
        detail = {}
        for model, trial in available.items():
            detail[model] = {
                "held_background_error": trial["held_background_error"],
                "held_increment": trial["held_error_reduction"].tolist(),
                "selected_phase": int(trial["phase"][0]),
            }
        rows[name] = {
            "free_pilot_sse": error_energy(data, pilot, cal),
            "shared_pilot_sse": error_energy(data, shared, cal),
            "nonpilot": detail,
        }
    held = data[~cal]
    return {"held_energy": float(np.vdot(held, held).real), "held_samples": int(len(held)),
            "calibration_samples": int(cal.sum()), "symbols": features["slots"], "models": rows}


def fold_channels(features, train):
    responses = np.stack([features[i]["basis_gains"][:, :-1] for i in train])
    # Complex channel shape common to the training frames, modulo one scalar
    # per frame. SVD is deterministic up to an irrelevant common phase.
    _, _, basis = np.linalg.svd(responses.transpose(1, 0, 2), full_matrices=False)
    return basis[:, 0]


def measure_window(iq, w, config, words, folds, slot_train, templates, frequencies):
    rate, cfo = w["rate_hz"], w["archived_cfo_hz"]
    edge = config["edge"]
    tones = visible_carriers(rate, cfo, templates.SUBCARRIER_SPACING_HZ, templates.qin_edge_pilot_indices(edge))
    waves = waveform_bank(words, tones[0])
    pilots = templates.qin_edge_pilot_symbols(edge)
    slot, cp = templates.OFDM_SYMBOL_DURATION_S, templates.CYCLIC_PREFIX_DURATION_S
    period = rate / templates.FRAME_RATE_HZ
    arms = {}
    for g, guard in enumerate(GUARDS):
        frames = []
        for f, gates in enumerate(w["frames"]):
            fractional = f * period
            shift = np.rint(fractional) - fractional
            origins = w["epoch_sample"] + fractional + np.array([0., shift, -shift])
            delta = gates[g]["actual_cfo_hz"]
            frames.append(frame_features(iq, rate, origins, frequencies, pilots, tones, cp, slot,
                                         cfo, delta, guard, slot_train))
        scored = []
        for train, held in (folds, folds[::-1]):
            channels = fold_channels(frames, train)
            for f in held:
                scored.append({"frame": int(f), "channel_training_frames": train.tolist(),
                               **score_frame(frames[f], channels, waves)})
        arms[str(guard)] = sorted(scored, key=lambda row: row["frame"])
    return {"key": w["key"], "carriers": tones[0].tolist(), "frame_period_samples": period, "guards": arms}


def main():
    require_run_context()
    sys.dont_write_bytecode = True
    inputs = {}
    origin = read(checked_artifact("v054-leo-cfo-confirm", "E056", "leo_cfo_aperture.json", inputs))
    gates = read(checked_artifact("v062-leo-pilot-phase-global", "E064", "leo_pilot_phase_transfer.json", inputs))
    tail = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    for rel, expected in origin["external_inputs_sha256"].items():
        if sha256(EXTERNAL_LEO_ROOT / rel) != expected:
            raise ValueError(f"LEO input differs from E056: {rel}")
    ds23 = EXTERNAL_LEO_ROOT / "reports/2026_10_19_fractional_frame_pilot_refinement"
    ds23_refs = {name: sha256(ds23 / name) for name in ("README.md", "source_manifest.json", "evidence.json")}
    sys.path[:0] = [str(EXTERNAL_LEO_ROOT / "src"), str(EXTERNAL_LEO_ROOT / "tools")]
    from leo.analysis.starlink import templates
    from dsx01_pilot_clock import edge_frequencies_hz

    rng = np.random.default_rng(SEED)
    pairs = np.arange(len(gates["windows"][0]["frames"])).reshape(-1, 2)
    partition = rng.permutation(len(pairs))
    folds = np.sort(pairs[partition[:3]].ravel()), np.sort(pairs[partition[3:]].ravel())
    word = np.asarray(tail["cycle_folds"][0]["signed_template"])
    words = np.stack([word, *[permute_lanes(word, rng) for _ in range(CONTROL_WORDS)]])
    windows = []
    for w in gates["windows"]:
        folder, name = w["key"].split("/")
        with np.load(EXTERNAL_LEO_ROOT / REPORT / folder / "selected_iq.npz", allow_pickle=False) as z:
            iq = z[name]
        if hashlib.sha256(iq.astype("<c8", copy=False).tobytes()).hexdigest() != w["iq_sha256"]:
            raise ValueError(f"LEO IQ differs from E064: {w['key']}")
        config = origin["windows"][w["key"]]
        hz = np.asarray(edge_frequencies_hz(config["edge"]))
        windows.append(measure_window(iq, w, config, words, folds, gates["calibration_blocks"], templates, hz))
        print(w["key"], "frames", len(w["frames"]), flush=True)
    result = {
        "schema": "starlink-leo-fractional-timing/v1", "seed": SEED, "controls": words.tolist(),
        "code_and_run_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
        "external_iq_and_receiver_inputs_sha256": origin["external_inputs_sha256"], "ds23_reference_sha256": ds23_refs,
        "whole_frame_folds": [x.tolist() for x in folds], "slot_calibration_blocks": gates["calibration_blocks"],
        "slot_held_blocks": gates["held_blocks"], "windows": windows,
    }
    out = experiment_output("E089")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leo_fractional_timing.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
