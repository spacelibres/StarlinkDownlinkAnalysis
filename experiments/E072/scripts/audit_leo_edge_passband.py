"""Digitally limit measured wideband IQ to the LEO upper-edge geometry."""

from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft, signal

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_ut_binary_tail import sha256
from code5_core import recover_generator
from layout_core import NONPILOT
from receiver_models import pn_reference
from ut_receiver_core import read_ci16


RATE = 250_000_000
WIDEBAND_RATE = 480_000_000
SPACING = 234_375
TARGET = 511
PILOTS = np.array([488, 490, 492, 494])
SEED = 2026112172
OFFSETS = (-250_000, -100_000, 0, 100_000, 250_000)
TAPERS = (0, 250_000, 1_000_000)
RAW = "data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin"


def lowpass_gain(frequency, sample_rate, taper):
    guard = sample_rate / 2 - abs(frequency)
    if taper == 0:
        return (guard > 0).astype(float)
    return np.sin(np.pi / 2 * np.clip(guard / taper, 0, 1)) ** 2


def cut_and_restore(mixed, sample_rate, taper):
    factor = int(RATE // sample_rate)
    if RATE != factor * sample_rate or len(mixed) % factor:
        raise ValueError("Capture segment and rate must align with the decimator")
    frequency = fft.fftfreq(len(mixed), 1 / RATE)
    filtered = fft.ifft(fft.fft(mixed) * lowpass_gain(frequency, sample_rate, taper))
    narrow = filtered[::factor]
    return signal.resample(narrow, len(mixed))


def erased_targets(observations, word_rows, checks):
    frames = defaultdict(list)
    for word, row in zip(observations, word_rows, strict=True):
        resources = row["start_resource"] + np.arange(len(word))
        erased = np.flatnonzero(NONPILOT[resources % len(NONPILOT)] == TARGET)
        if not len(erased):
            continue
        kept = word.copy()
        kept[erased] = 0
        solution = recover_generator(checks[:, erased], ((checks @ kept) & 1)[:, None])
        if solution["generator"] is None:
            raise ValueError(f"Target erasure not resolved at {row['candidate_index']}/{row['word']}")
        recovered = solution["generator"].ravel()
        kept[erased] = recovered
        if np.any((checks @ kept) & 1):
            raise ValueError("Restored reference does not pass the frozen checks")
        for position, bit in zip(resources[erased], recovered, strict=True):
            frames[row["candidate_index"]].append((int(position), int(bit)))
    return frames


def symbol_receipt(raw, frame, positions, conditions, target_baseband):
    slots = 2 + np.asarray([position for position, _ in positions]) // len(NONPILOT)
    labels = np.array([bit for _, bit in positions])
    scale = frame["sampling_scale"]
    epoch = frame["pss"]["epoch"]
    cfo = frame["cfo_hz"]
    start = epoch + (slots * 2112 + 64) * scale
    raw_start = start * RATE / WIDEBAND_RATE
    first = max(0, int(min(raw_start)) - 40_000)
    stop = min(len(raw), int(max(raw_start)) + 42_000)
    stop -= (stop - first) % 10
    local = raw[first:stop]
    clocks = np.arange(first, stop) / RATE
    carriers = np.r_[PILOTS, TARGET]
    width = int(np.floor(2048 * scale * RATE / WIDEBAND_RATE)) - 1
    samples = np.ceil(raw_start).astype(int)[:, None] + np.arange(width) - first
    if samples.min() < 0 or samples.max() >= len(local):
        raise ValueError("Projection would use samples outside the recording segment")
    origins = start / WIDEBAND_RATE
    relative = clocks[samples] - origins[:, None]
    tones = np.exp(-2j * np.pi * (relative[:, :, None] * (carriers * SPACING)[None, None, :] / scale
                                  + cfo * clocks[samples, None]))
    pn = np.column_stack([pn_reference(int(carrier), slots) for carrier in carriers])

    def project(values):
        return np.einsum("ij,ijk->ik", values[samples], tones) / (width * pn)

    outputs = {"raw": project(local)}
    for offset, taper in conditions:
        tune = cfo + TARGET * SPACING / scale - target_baseband - offset
        rotor = np.exp(-2j * np.pi * (tune / RATE) * np.arange(len(local)))
        restored = cut_and_restore(local * rotor, 25_000_000, taper) * rotor.conj()
        outputs[f"{offset}/{taper}"] = project(restored)
    return labels, outputs


def compare(reference, candidate):
    dot = np.vdot(reference, candidate)
    energy = np.sum(abs(reference) ** 2)
    total = energy * np.sum(abs(candidate) ** 2)
    return {"transfer": float(abs(dot) / energy), "coherence": float(abs(dot) / np.sqrt(total)),
            "relative_error": float(np.sum(abs(candidate - reference) ** 2) / energy)}


def assess(reference, candidate, signs, phase):
    return {"target": compare(reference[:, -1], candidate[:, -1]),
            "pilots": compare(reference[:, :-1], candidate[:, :-1]),
            "sign_matches": int(np.count_nonzero((signs * (candidate[:, -1] * phase).real) > 0)),
            "positions": int(len(signs))}


def early_to_late(rows, held, names):
    anchor_count = 2
    eligible = [frame for frame in held if len(rows[frame][0]) > anchor_count]
    phases = {}
    for frame in eligible:
        labels, values = rows[frame]
        signs = 1 - 2 * labels[:anchor_count]
        phases[frame] = {
            name: np.exp(-1j * np.angle(np.dot(signs, values[name][:anchor_count, -1])))
            for name in names
        }
    scores = {}
    for name in names:
        result = []
        for frame, donor in zip(eligible, eligible[1:] + eligible[:1], strict=True):
            labels, values = rows[frame]
            observed = values[name][anchor_count:, -1]
            signs = 1 - 2 * labels[anchor_count:]
            counts = {
                key: int(np.count_nonzero((observed * rotor).real * signs > 0))
                for key, rotor in {
                    "same_frame_raw_phase": phases[frame]["raw"],
                    "same_frame_narrow_phase": phases[frame][name],
                    "wrong_frame_narrow_phase": phases[donor][name],
                }.items()
            }
            result.append({"frame": frame, "wrong_frame": donor, "positions": len(signs), **counts})
        scores[name] = {
            "counts": {key: sum(row[key] for row in result) for key in (
                "positions", "same_frame_raw_phase", "same_frame_narrow_phase", "wrong_frame_narrow_phase")},
            "by_frame": result,
        }
    return {"per_frame_anchor_count": anchor_count, "scores": scores}


def main():
    require_run_context()
    source = ROOT / RAW
    receipt_path = ROOT / "runs/v022-ut-replay/artifacts/experiments/E014/results/ut_transfer.json"
    words_path = receipt_path.with_name("ut_observations.npz")
    checks_path = ROOT / "runs/v011-sparse-code5/artifacts/experiments/E013/results/sparse_code5_arrays.npz"
    leo_path = ROOT / "runs/v069-leo-recording-coverage/artifacts/experiments/E071/results/leo_recording_coverage.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    hashes = {str(path.relative_to(ROOT).as_posix()): sha256(path)
              for path in (source, receipt_path, words_path, checks_path, leo_path)}
    if hashes[RAW] != receipt["source_sha256"]:
        raise ValueError("Raw IQ differs from E014 input")
    with np.load(words_path, allow_pickle=False) as words, np.load(checks_path, allow_pickle=False) as checks:
        positions = erased_targets(words["hard_code5_words"], receipt["words"], checks["check_matrix"])
    for records in positions.values():
        records.sort()
    ids = sorted(positions)
    training = set(np.random.default_rng(SEED).permutation(ids)[:len(ids) // 2].tolist())
    leo = json.loads(leo_path.read_text(encoding="utf-8"))["separate_native_glrt_capture"]
    target = leo["target_baseband_hz"]
    conditions = list(product(OFFSETS, TAPERS))
    candidates = {row["candidate_index"]: row for row in receipt["candidates"]}
    raw = read_ci16(source)
    rows = {frame: symbol_receipt(raw, candidates[frame], positions[frame], conditions, target)
            for frame in ids}
    signed_reference = sum(np.dot(1 - 2 * rows[frame][0], rows[frame][1]["raw"][:, -1])
                           for frame in training)
    phase = np.exp(-1j * np.angle(signed_reference))
    held = [frame for frame in ids if frame not in training]
    names = list(rows[ids[0]][1])
    held_raw = np.vstack([rows[frame][1]["raw"] for frame in held])
    signs = 1 - 2 * np.concatenate([rows[frame][0] for frame in held])
    scores = {name: assess(held_raw, np.vstack([rows[frame][1][name] for frame in held]), signs, phase)
              for name in names}
    output = {
        "schema": "starlink-leo-edge-passband/v1",
        "inputs_sha256": hashes,
        "seed": SEED,
        "training_frames": sorted(training),
        "holdout_frames": held,
        "held_scores_by_cfo_hz_and_taper_hz": scores,
        "target_baseband_hz": target,
        "target_phase_calibration_rad": float(np.angle(phase)),
        "followup_early_to_late": early_to_late(rows, held, names),
        "frames": {
            str(frame): {
                "references": positions[frame],
                "scores": {name: assess(values["raw"], data, 1 - 2 * labels, phase)
                           for name, data in values.items()},
                "target_coefficients": {name: np.column_stack([data[:, -1].real, data[:, -1].imag]).tolist()
                                        for name, data in values.items()},
            }
            for frame, (labels, values) in rows.items()
        },
    }
    folder = experiment_output("E072")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leo_edge_passband.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    for name, score in scores.items():
        print(name, score)


if __name__ == "__main__":
    main()
