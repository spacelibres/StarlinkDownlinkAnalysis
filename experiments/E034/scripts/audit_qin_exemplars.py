"""Replay frozen Starlink headers and Code 21 on the Qin exemplar archive."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code28_reencoding import fft_binary_parities
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, equalize_grid, reference_gain
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from ut_receiver_core import decode_variable_pdu
from validate_code21 import coordinates

SNAPSHOT = "v032-qin-snapshot"
CALIBRATION_FRAME = 991
WORD_SYMBOLS = 4096
ALIGN_TOLERANCE = 1e-6


def header_key(first, second):
    if first.get("crc_ok") is True and second.get("crc_ok") is True:
        return tuple(first["words_hex"] + second["words_hex"])
    return None


def consistent_clock_pairs(pairs, tol):
    if not pairs:
        raise ValueError("No shared header fingerprints")
    shifts = np.array([pair["toa_offset_s"] for pair in pairs])
    totals = (abs(shifts[:, None] - shifts) <= tol).sum(axis=1)
    center = shifts[np.argmax(totals)]
    group = [pair for pair in pairs if abs(pair["toa_offset_s"] - center) <= tol]
    if len({pair["qin_frame"] for pair in group}) != len(group):
        raise ValueError("Frequency-domain archive frame match is ambiguous")
    if len({pair["ut_candidate"] for pair in group}) != len(group):
        raise ValueError("IQ candidate frame match is ambiguous")
    offset = float(np.median([pair["toa_offset_s"] for pair in group]))
    return offset, [{**pair, "residual_s": pair["toa_offset_s"] - offset} for pair in group]


def remove_header_gain(values, signs):
    flat = values.ravel()
    return flat / reference_gain(flat[:48], signs)


def frame_headers(flat):
    h1 = decode_variable_pdu(flat[:228], "h1")
    h2 = {}
    start = None
    if h1.get("crc_ok") is True:
        entries = h1["layout20"]["entries"]
        if (h1["layout20"]["packing_valid"] and len(entries) == 1
                and entries[0]["mcs_candidate"] == 5 and entries[0]["quantity"] > 0):
            start = 228 + 2048 * entries[0]["quantity"]
            h2 = decode_variable_pdu(flat[start:start+798], "h2")
    return h1, h2, start


def describe_frame(frame, toa, sinr, flat, ut_by_key):
    h1, h2, start = frame_headers(flat)
    fingerprint = header_key(h1, h2)
    pairs = [{"qin_frame": int(frame), "ut_candidate": c["candidate_index"],
              "toa_offset_s": float(toa) - c["pss"]["epoch"]/480e6}
             for c in ut_by_key.get(fingerprint, [])]
    header_objects = [{key: value for key, value in h.items() if key not in ("bits", "reencoded")} for h in (h1, h2)]
    rec = {"frame": int(frame), "toa_s": float(toa), "sinr_db": float(sinr), "h1": header_objects[0],
           "h2": header_objects[1], "h2_start": start}
    return rec, pairs


def word_bits_at(flat, start, count, orientation, move):
    point = start + move
    stop = point + count * WORD_SYMBOLS
    if point < 0 or stop > len(flat):
        raise ValueError("Code 21 allocation is outside retained samples")
    return coordinates(flat[point:stop].reshape(count, WORD_SYMBOLS), *orientation)


def parity_counts(bits, generator, shift=0):
    predicted, residual = fft_binary_parities(bits[:, :5120].reshape(-1, 10, 512), generator)
    if shift:
        predicted = np.roll(predicted, shift, axis=1)
    return np.count_nonzero(predicted != bits[:, 5120:], axis=1).tolist(), residual


def code21_transfer(meta, soft, pn, signs, sss, generator):
    observations = {}
    slot_ids = np.arange(2, soft["symbols"].shape[1] + 1)
    for frame, samples in zip(soft["frame"], soft["symbols"], strict=True):
        m = meta[int(frame)]
        h2 = m["h2"]
        if h2.get("crc_ok") is not True or not h2["layout20"]["packing_valid"]:
            continue
        first = h2["layout20"]["entries"][0]
        if first["mcs_candidate"] != 21:
            continue
        start = m["h2_start"] + h2["blocks"] * 114
        direct = remove_header_gain(samples[1:, NONPILOT % 1024] * pn[:len(slot_ids)], signs)
        calibrated = equalize_grid(samples[1:, LOADED % 1024], samples[0, LOADED % 1024], sss, slot_ids)
        pilot = remove_header_gain(calibrated["nonpilot"], signs)
        observations[int(frame)] = (start, first["quantity"], {"direct": direct, "pilot": pilot},
                                    calibrated["reserved_pilot_evm"])
    orientations = [(direction, phase) for direction in (1, -1) for phase in range(4)]
    start, count, data, _ = observations[CALIBRATION_FRAME]
    models = {}
    for front_end, column in data.items():
        trials = [(orientation, parity_counts(word_bits_at(column, start, count, orientation, 0), generator)[0])
                  for orientation in orientations]
        selected = min(trials, key=lambda trial: sum(trial[1]))[0]
        models[front_end] = {"orientation": list(selected), "calibration_attempts": [
            {"orientation": list(orientation), "errors_by_word": errors} for orientation, errors in trials]}
    rows = []
    for frame, (start, count, branches, evm) in observations.items():
        metrics = {}
        for name, raw in branches.items():
            orientation = models[name]["orientation"]
            word_bits = word_bits_at(raw, start, count, orientation, 0)
            errors, numerical = parity_counts(word_bits, generator)
            metrics[name] = {"errors_by_word": errors, "max_fft_roundoff": numerical,
                             "raw_legacy_phase": parity_counts(word_bits_at(raw, start, count, (1, 0), 0), generator)[0],
                             "offset_minus1": parity_counts(word_bits_at(raw, start, count, orientation, -1), generator)[0],
                             "offset_plus1": parity_counts(word_bits_at(raw, start, count, orientation, 1), generator)[0],
                             "rotated_parity": parity_counts(word_bits, generator, shift=1)[0]}
        rows.append({"frame": frame, "calibration": frame == CALIBRATION_FRAME, "start_resource": start, "words": count,
                     "reserved_pilot_evm": evm, "branches": metrics})
    return models, rows


def main():
    require_run_context()
    inputs = {}
    source = read(checked_artifact(SNAPSHOT, "E034", "qin_snapshot_source.json", inputs))
    prior = read(checked_artifact("v022-ut-replay", "E014", "ut_transfer.json", inputs))
    numeric = checked_artifact("v014-code21-repeat", "E016", "code21_observations.npz", inputs)
    sss_path = ROOT / "references/qin/sssVec.mat"
    inputs[sss_path] = sha256(sss_path)
    sss = loadmat(sss_path)["sssVecFull"].ravel()[LOADED % 1024]
    p = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    known = np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)]
    signs = 1 - 2 * encode_block(known)[:48].astype(float)
    by_header = {}
    for case in prior["candidates"]:
        k = header_key(case.get("h1", {}), case.get("h2", {}))
        if k:
            by_header.setdefault(k, []).append(case)
    frames, pairs = [], []
    for group in source["hard_groups"]:
        path = checked_artifact(SNAPSHOT, "E034", f"qin_hard_{group}.npz", inputs)
        with np.load(path, allow_pickle=False) as v:
            for frame, toa, sinr, samples in zip(v["frame"], v["toa_s"], v["sinr_db"], v["symbols"], strict=True):
                flat = remove_header_gain(samples[1:, NONPILOT % 1024] * p, signs)
                row, overlap = describe_frame(frame, toa, sinr, flat, by_header)
                frames.append(row)
                pairs.extend(overlap)
    offset, aligned = consistent_clock_pairs(pairs, ALIGN_TOLERANCE)
    soft_file = checked_artifact(SNAPSHOT, "E034", "qin_soft_988.npz", inputs)
    with np.load(numeric, allow_pickle=False) as gen, np.load(soft_file, allow_pickle=False) as soft:
        generator = gen["parity_generator"][::512].reshape(10, 6, 512)
        models, c21 = code21_transfer({row["frame"]: row for row in frames}, soft, p, signs, sss, generator)
    descriptors = Counter(
        item["mcs_candidate"] for frame in frames if frame["h2"].get("crc_ok") is True
        for item in frame["h2"]["layout20"]["entries"]
    )
    result = {"schema": "starlink-qin-exemplar-transfer/v1",
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "overlap": {"offset_s": offset, "tolerance_s": ALIGN_TOLERANCE, "pairs": aligned},
              "frame_count": len(frames), "headers": frames,
              "crc_accepted_h1": sum(x["h1"].get("crc_ok") is True for x in frames),
              "crc_accepted_h2": sum(x["h2"].get("crc_ok") is True for x in frames),
              "descriptors_by_code": dict(sorted(descriptors.items())),
              "code21_calibration_frame": CALIBRATION_FRAME, "code21_orientation_models": models, "code21": c21,
              "scope": "Separate RF capture from the legacy 24 ms IQ. The Qin corpus includes the already used 250 MS/s "
                       "recording and one satellite observation session, not independent repetitions per frame. "
                       "Header and Code 21 coordinates are receiver-relative. No payload fields inferred."}
    out = experiment_output("E034")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qin_exemplar_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("CRC", result["crc_accepted_h1"], result["crc_accepted_h2"], "of", len(frames), "overlap", len(aligned), flush=True)
    for branch in models:
        held = [e for row in c21 if not row["calibration"] for e in row["branches"][branch]["errors_by_word"]]
        print(branch, sum(e == 0 for e in held), "exact held words of", len(held), flush=True)


if __name__ == "__main__":
    main()
