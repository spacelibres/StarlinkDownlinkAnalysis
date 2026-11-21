"""Separate slot quantizer magnitude loss from code-coordinate differences."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_pn_padding_starts import CacheOnly, demap_exact, lattice_model
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


SEED = 2026112175
RESOURCES_PER_SLOT = len(NONPILOT)
CASES = {66: 12288, 70: 12800}
SOURCE_RUN = "v072-padding-parity-final"
SOFT_SUBSET = {(630, 631), (640, 641)}


def corner_slots(left, right):
    left_outer = np.isclose(abs(left), 1, atol=2e-6)
    right_outer = np.isclose(abs(right), 1, atol=2e-6)
    return (left_outer.all(axis=1) & ((~right_outer).mean(axis=1) >= .25)) | (
        right_outer.all(axis=1) & ((~left_outer).mean(axis=1) >= .25))


def calibration_gain(measured, reference, training):
    design = reference[training]
    return np.vdot(design, measured[training]) / np.vdot(design, design)


def holdout_error(measured, reference, training):
    scale = calibration_gain(measured, reference, training)
    test = measured[~training]
    return float(np.sum(abs(test - scale * reference[~training]) ** 2) / np.sum(abs(test) ** 2))


def compared_word(hard, masks, resource, edge, lost, amplitude):
    span = slice(resource, resource + 4096)
    cells = (np.arange(resource, span.stop) // RESOURCES_PER_SLOT)
    old = np.repeat((masks[0][span] & masks[1][span])[:, None], hard[0].shape[-1], axis=1)
    new = old & ~(lost[cells, None] & amplitude[None, :])
    diff = (hard[0][span] != hard[1][span])
    raw_changes = (diff & old).ravel()
    corrected_changes = (diff & new).ravel()
    system_by_slot = np.bincount(cells.repeat(4)[:edge], raw_changes[:edge], minlength=len(lost))
    return {
        "raw_system_observed": int(old.ravel()[:edge].sum()),
        "raw_system_differences": int(raw_changes[:edge].sum()),
        "raw_parity_differences": int(raw_changes[edge:].sum()),
        "erased_system_bits": int((old & ~new).ravel()[:edge].sum()),
        "erased_parity_bits": int((old & ~new).ravel()[edge:].sum()),
        "remaining_system_differences": int(corrected_changes[:edge].sum()),
        "remaining_parity_differences": int(corrected_changes[edge:].sum()),
        "dominant_system_slot": int(system_by_slot.argmax()) + 2,
        "dominant_system_slot_errors": int(system_by_slot.max()),
        "masked_slots": (np.unique(cells[lost[cells]]) + 2).tolist(),
    }


def magnitude_levels(values):
    return np.column_stack([abs(values.real), abs(values.imag)]) > np.sqrt(2) / 3


def test_soft(dataset, frames, slot, four, normal, rotation, training, controls, decode):
    sample = np.linspace(0, RESOURCES_PER_SLOT - 1, 64, dtype=int)
    bins = NONPILOT[sample] % 1024
    order = bins.argsort()
    reference = normal[slot - 2, sample]
    corners = (np.sign(reference.real) + 1j * np.sign(reference.imag)) / np.sqrt(2)
    preserved_sign_amplitudes = [np.sign(reference.real) * abs(reference.real[p])
                                 + 1j * np.sign(reference.imag) * abs(reference.imag[p])
                                 for p in controls]
    tests = []
    for frame in frames:
        measured = as_complex(dataset[frame, slot - 1, bins[order]])[order.argsort()]
        measured = measured * rotation[slot - 2, sample]
        full = holdout_error(measured, reference, training)
        corner = holdout_error(measured, corners, training)
        wrong = [holdout_error(measured, row, training) for row in preserved_sign_amplitudes]
        result = {
            "frame": frame, "hard_constellation": 4 if frame == four else 16,
            "reference_16qam_nmse": full, "corner_nmse": corner, "shuffled_amplitude_nmse": wrong,
            "measured_iq_after_pn": [[float(z.real), float(z.imag)] for z in measured],
        }
        if decode:
            gain = calibration_gain(measured, reference, training)
            truth = magnitude_levels(reference[~training])
            predicted = magnitude_levels(measured[~training] / gain)
            result["held_magnitude_decisions"] = {
                "positions": int(truth.size), "soft_level_agreements": int(np.count_nonzero(predicted == truth)),
                "all_outer_agreements": int(truth.sum()),
                "gain": [float(gain.real), float(gain.imag)],
                "per_carrier_axis_agreement": (predicted == truth).tolist(),
            }
        tests.append(result)
    return {"sample_resource_positions": sample.tolist(),
            "reference_16qam_iq": [[float(z.real), float(z.imag)] for z in reference], "tests": tests}


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--magnitude-decisions"]):
        raise ValueError("Only --magnitude-decisions is supported")
    decode = bool(sys.argv[1:])
    inputs = {}
    repeats = read(checked_artifact(SOURCE_RUN, "E074", "padding_parity_transfer.json", inputs))
    with np.load(checked_artifact("v026-ut-qam-extract-repeat", "E028", "ut_qam_symbols.npz", inputs),
                 allow_pickle=False) as data:
        points = data["constellation_points"]
        labels = data["frozen_gray_mapping"]
    model = lattice_model(points, labels)
    amplitude = np.ptp(labels[np.isclose(abs(points), abs(points).max())], axis=0) == 0
    if amplitude.sum() != 2:
        raise ValueError("Frozen 16QAM mapping has unexpected magnitude coordinates")
    slots = np.arange(2, 302)
    rotation = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    reference_signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rng = np.random.default_rng(SEED)
    training_groups = sorted(rng.permutation(8)[:4].tolist())
    train = np.repeat(np.isin(np.arange(8), training_groups), 8)
    controls = [rng.permutation(64) for _ in range(99)]
    pairs = []
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as file:
        source = file["yDataDec"]
        for pair in repeats["nearby_diagnostic"]["same_header_pairs"]:
            targets = [row for row in pair["repeated_words"] if row["label"] in CASES]
            if not targets:
                continue
            frames = pair["frames"]
            norms = [remove_header_gain(as_complex(source[frame, 1:, :])[..., NONPILOT % 1024] * rotation,
                                        reference_signs).reshape(300, RESOURCES_PER_SLOT) for frame in frames]
            decoded = [demap_exact(norm.ravel(), model) for norm in norms]
            hard, masks = [d[0] for d in decoded], [d[1] for d in decoded]
            full_slot_visible = np.logical_and.reduce(
                [mask.reshape(300, RESOURCES_PER_SLOT).mean(axis=1) >= .90 for mask in masks])
            lost = corner_slots(*norms) & full_slot_visible
            results, soft = [], {}
            for target in targets:
                edge = CASES[target["label"]]
                row = compared_word(hard, masks, target["resource"], edge, lost, amplitude)
                if row["raw_system_differences"] != target["systematic_disagreements"]:
                    raise ValueError("Hard differences diverged from E074")
                if row["raw_system_differences"] >= 1000:
                    slot = row["dominant_system_slot"]
                    order = [int(file[file["modEst"][frame, slot - 1]].size) for frame in frames]
                    row["author_modulation_orders"] = order
                    if tuple(frames) in SOFT_SUBSET and sorted(order) == [4, 16] and slot not in soft:
                        four, sixteen = order.index(4), order.index(16)
                        soft[slot] = test_soft(file["yDataSoft"], frames, slot, frames[four],
                                               norms[sixteen], rotation, train, controls, decode)
                results.append({**target, **row})
            pairs.append({"frames": frames, "words": results, "soft_by_slot": soft})
    previous_pages = {row["bytes"][0]: row["sha256"] for row in repeats["pages"]}
    if any(previous_pages[start] != row["sha256"] for start, row in reader.receipts.items()
           if start in previous_pages):
        raise ValueError("Archive ranges reused from E074 changed")
    output = {
        "schema": "starlink-qin-amplitude-erasure/v1", "seeds": [SEED],
        "magnitude_lanes_zero_based": np.flatnonzero(amplitude).tolist(),
        "soft_training_groups": training_groups,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                   "pages": [reader.receipts[page] for page in sorted(reader.receipts)]},
        "pairs": pairs,
    }
    folder = experiment_output("E075")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "qin_amplitude_erasure.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                                       encoding="utf-8")
    for pair in pairs:
        print(pair["frames"], "words", len(pair["words"]), "soft slots", sorted(pair["soft_by_slot"]))


if __name__ == "__main__":
    main()
