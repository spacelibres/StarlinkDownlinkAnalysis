"""Decode measured MCS 70/88 with the virtual block erased, then re-encode."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross32_partial_decode import raw_inputs
from audit_mcs66_partial_decoder import GUARD_BITS, RAW_VARIANCE, constellation_llr
from audit_qam_pn15 import collect, equality_predictions
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs70_full_code import BIT_LENGTH, EXTENDED_BITS, SYSTEMATIC
from validate_mcs66_full_code import decode_words, invert_binary, remodulate


MASK_SEED = 20261121131


def encode_transmitted(bits, systematic_rows, inverse):
    systematic = bits[:, :SYSTEMATIC]
    rhs = np.column_stack([np.bitwise_xor.reduce(systematic[:, row], axis=1) for row in systematic_rows])
    values = [int.from_bytes(row.tobytes(), "little") for row in np.packbits(rhs, axis=1, bitorder="little")]
    parity = np.array([[(row & v).bit_count() % 2 for row in inverse] for v in values], dtype=bool)
    return np.column_stack((systematic, parity))


def measured70(artifact, reserved):
    run = "v026-ut-qam-extract-repeat"
    meta = read(artifact(run, "E028", "ut_qam_extraction.json"))
    observations, points, mapping = collect(artifact(run, "E028", "ut_qam_symbols.npz"), meta)
    if meta["reserved_frames"] != reserved:
        raise ValueError("Raw capture holdout differs")
    bits, _, samples, rows = observations[70]
    frames = np.array([row['frame'] for row in rows])
    known = np.zeros_like(bits, dtype=bool)
    evidence = read(artifact("v027-qam-pn15-final", "E029", "qam_pn15_audit.json"))
    for receipt in evidence["words"]:
        if receipt["format"] != 70 or not receipt["match"]["accepted"]:
            continue
        i = receipt["candidate_index"]
        if any(rows[i][key] != receipt[key] for key in ("frame", "resource_start", "last_word_of_allocation")):
            raise ValueError("PN15 receipt does not match the measured terminal word")
        region = receipt["match"]
        known[i, region["start"] + GUARD_BITS:min(SYSTEMATIC, region["stop"] - GUARD_BITS)] = True
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), evidence["fit"]["winner"]).astype(bool)
    take = np.isin(frames, reserved)
    prior = constellation_llr(samples[take], points, mapping, RAW_VARIANCE).reshape(int(take.sum()), BIT_LENGTH)
    return samples[take], points, mapping, frames[take], prior.astype(np.float32), bits[take], known[take], pn


def measured88(artifact, labels, reserved):
    path = artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    points = np.array([complex(r["i"], r["q"]) for r in labels["cross32_mapping"]]) / np.sqrt(20)
    mapping = labels_as_bits(np.array([r["label"] for r in labels["cross32_mapping"]]), 5)
    with np.load(path, allow_pickle=False) as z:
        frames, prior, bits, known = raw_inputs(88, z, labels, SYSTEMATIC)
        samples = z["mcs88_symbols"]
    take = np.isin(frames, reserved)
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    return samples[take], points, mapping, frames[take], prior[take], bits[take], known[take], pn


def decode_capture(prior, support):
    # A virtual parity block has no channel LLR. The one extra column is a known zero
    # that brings both row weights to 15 for the existing rectangular decoder.
    lifted = np.column_stack((prior, np.zeros((len(prior), EXTENDED_BITS - BIT_LENGTH), dtype=prior.dtype),
                              np.full(len(prior), np.inf, dtype=prior.dtype)))
    bits, rounds = decode_words(lifted, support)
    if np.any(bits[:, -1]):
        raise ValueError("The fixed-zero padding changed sign")
    return bits[:, :BIT_LENGTH], rounds


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model = artifact("v129-mcs70-full-code", "E131", "mcs70_full_checks.npz")
    with np.load(model, allow_pickle=False) as z:
        families = [z[f"extended_family_{i}"] for i in range(8)]
        projected = np.unpackbits(z["projected_matrix"], axis=1)
    support = np.vstack([row if row.shape[1] == 15 else np.column_stack((row, np.full(len(row), EXTENDED_BITS)))
                         for row in families])
    by_row = np.nonzero(projected[:, :SYSTEMATIC])
    systematic_rows = np.split(by_row[1], np.searchsorted(by_row[0], np.arange(1, len(projected))))
    inverse = invert_binary(projected[:, SYSTEMATIC:])
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    reserved = labels["frame_split"]["reserved"]
    packed, report = {}, {}
    for code, data in ((70, measured70(artifact, reserved)), (88, measured88(artifact, labels, reserved))):
        samples, points, mapping, frames, prior, hard, known, pn = data
        width = mapping.shape[1]
        parity_symbols = np.arange((SYSTEMATIC + width - 1) // width, BIT_LENGTH // width)
        masked = np.sort(np.random.default_rng(MASK_SEED).choice(parity_symbols, len(parity_symbols) // 16, replace=False))
        target_bits = (masked[:, None] * width + np.arange(width)).ravel()
        withheld = prior.copy()
        withheld[:, target_bits] = 0.
        base, rounds = decode_capture(prior, support)
        erasures, masked_rounds = decode_capture(withheld, support)
        if not np.all(rounds > 0) or not np.all(masked_rounds > 0) or not np.array_equal(base, erasures):
            raise ValueError(f"Measured ordinary and erased decodes disagree for MCS {code}")
        encoded = encode_transmitted(erasures, systematic_rows, inverse)
        if not np.array_equal(encoded, erasures):
            raise ValueError("Independent transmitted-domain encoder does not match")
        predicted = remodulate(encoded, points, mapping)[:, masked]
        wrong = encoded.copy()
        wrong[:, SYSTEMATIC:] = np.roll(wrong[:, SYSTEMATIC:].reshape(len(wrong), -1, 512), 1, axis=-1).reshape(
            len(wrong), BIT_LENGTH - SYSTEMATIC)
        control = remodulate(wrong, points, mapping)[:, masked]
        observed = samples[:, masked]
        norm = (abs(observed) ** 2).sum(axis=1)
        errors = (abs(predicted - observed) ** 2).sum(axis=1) / norm
        control_errors = (abs(control - observed) ** 2).sum(axis=1) / norm
        pn_after = ((base != pn) & known).sum(axis=1)
        if not known.any() or pn_after.any() or np.any(errors >= control_errors):
            raise ValueError(f"PN15 or waveform comparison failed for MCS {code}")
        report[code] = {"words": len(base), "frames": np.unique(frames).tolist(), "masked_modulation_symbols": masked.tolist(),
                        "rows": [{"frame": int(frame), "ordinary_first_pass_rounds": int(rounds[i]),
                                  "masked_first_pass_rounds": int(masked_rounds[i]), "pn_checked": int(known[i].sum()),
                                  "pn_hard_differences": int(((hard[i] != pn) & known[i]).sum()),
                                  "pn_decoded_differences": int(pn_after[i]), "withheld_waveform_nmse": float(errors[i]),
                                  "shifted_parity_nmse": float(control_errors[i])} for i, frame in enumerate(frames)]}
        for label, bits in (("ordinary", base), ("masked", erasures)):
            packed[f"mcs{code}_{label}_bits"] = np.packbits(bits, axis=1)
        print(code, len(base), "unchanged words; PN15", int(known.sum()), "NMSE", errors.mean(), control_errors.mean(), flush=True)

    result = {"schema": "starlink-mcs70-full-code-validation/v1", "mask_seed": MASK_SEED, "reserved_frames": reserved,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}, "measurements": report}
    path = experiment_output("E131")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "mcs70_full_decoded_words.npz", **packed)
    (path / "mcs70_full_decoder_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
