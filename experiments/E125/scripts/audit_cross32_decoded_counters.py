"""Replay fixed MCS 85 field offsets and ablate the observed carry byte."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross32_partial_decode import BIT_LENGTH, LIFT, decode, raw_inputs
from audit_mcs85_counter import tagged_pairs
from audit_mcs85_records import system_bytes
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


SYSTEMATIC = 12288


def fields(raw):
    return {"tag": raw[4:6].hex(), "counter": int.from_bytes(raw[6:8], "big")}


def pair_fields(frame, streams, lengths):
    candidates = {row["byte_offset"]: fields(bytes.fromhex(row["observed_hex"])) for row in frame["records"]}
    pairs = []
    for pair in tagged_pairs(frame["records"], lengths):
        offsets = [pair["from"], pair["to"]]
        modes = {"nearest": [candidates[p] for p in offsets]}
        for name, source in streams.items():
            modes[name] = [fields(source[p:p + 8].tobytes()) for p in offsets]
        pairs.append({**pair, "frame": frame["frame"], "observations": {
            name: {"ends": ends, "step": (ends[1]["counter"] - ends[0]["counter"]) % 65536}
            for name, ends in modes.items()
        }})
    return pairs


def score(pairs):
    result = {}
    for name in ("nearest", "decoded", "shifted"):
        counts = Counter()
        for pair in pairs:
            data = pair["observations"][name]
            counts["both_tags_match_spacing"] += all(r["tag"] == pair["tag"] for r in data["ends"])
            step = data["step"]
            counts["plus1"] += step == 1
            counts["plus1_to_4"] += 1 <= step <= 4
            counts["minus1_to_4"] += 65532 <= step <= 65535
        result[name] = dict(counts)
    return result


def masked_carry(prior, frames, support, pair, pn, chosen):
    byte_offset = pair["to"] + 6
    coordinates = byte_offset * 8 + np.arange(8)
    words, positions = np.divmod(coordinates, SYSTEMATIC)
    result = {}
    source_rows = np.flatnonzero(frames == pair["frame"])
    for tag, rows in (("checks", support),
                      ("shifted", np.where(support < SYSTEMATIC,
                                           support // LIFT * LIFT + (support + 1) % LIFT, support))):
        result[tag] = {}
        for method in ("min_sum", "sum_product"):
            llr = np.empty(8, dtype=np.float32)
            for word in np.unique(words):
                use = words == word
                layout = positions[use]
                signal = prior[source_rows[word]:source_rows[word] + 1].copy()
                signal[0, layout] = 0.0
                value = decode(signal, rows, method, chosen[1])[0, layout]
                llr[use] = value * (1 - 2 * pn[layout].astype(float))
            result[tag][method] = {"descrambled_llr": llr.tolist(),
                                  "byte": int(np.packbits(llr < 0, bitorder="little")[0])
                                  if np.all(abs(llr) > 1e-8) else None}
    return {"tested_byte_offset": byte_offset, "word_indices": words.tolist(),
            "serial_positions": positions.tolist(), "methods": result}


def archive_binding(pair, binding, records):
    by_raw = {row["ut_candidate"]: row["qin_frame"] for row in binding}
    archive = by_raw.get(pair["frame"])
    wanted = (pair["from"], pair["to"])
    found = {}
    for row in records["rows"]:
        if row["frame"] == archive:
            for r in row["records"]:
                if r["byte_offset"] in wanted:
                    found[r["byte_offset"]] = {"observed_hex": r["observed_hex"],
                                               "observed_mask_hex": r["observed_mask_hex"]}
    return {"qin_frame": archive, "tag_positions_detected": [found.get(k) for k in wanted]}


def main():
    require_run_context()
    receipts = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, receipts)

    record = read(art("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    prior_run = "v123-cross32-partial-decode"
    decode_record = read(art(prior_run, "E125", "cross32_partial_decode.json"))["formats"]["85"]["raw250"]
    frames = np.asarray(decode_record["frames"])
    labels = read(art("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    if record["frames_split"] != decode_record["raw_group_assignment"] or record["pn_model"] != labels["frozen_pn_model"]:
        raise ValueError("MCS 85 field receipts disagree")
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), record["pn_model"])
    with np.load(art(prior_run, "E125", "cross32_partial_decisions.npz"), allow_pickle=False) as packed:
        decoded = {mode: np.unpackbits(packed[f"85_{mode}_bits"], axis=1) for mode in ("decoded", "shifted")}
    pairs = []
    for r in record["frame_results"]:
        if r["frame"] in record["frames_split"]["reserved"]:
            streams = {mode: system_bytes(bits[frames == r["frame"]], SYSTEMATIC, pn).ravel()
                       for mode, bits in decoded.items()}
            pairs.extend(pair_fields(r, streams, record["fitted_distances_by_exact_label"]))
    candidates = [pair for pair in pairs if pair["observations"]["decoded"]["step"] == 1
                  and pair["observations"]["decoded"]["ends"][0]["counter"] % 256 == 255]
    with np.load(art("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz"), allow_pickle=False) as src:
        actual_frames, prior, _, _ = raw_inputs(85, src, labels, SYSTEMATIC)
    if not np.array_equal(frames, actual_frames):
        raise ValueError("Bit and waveform frame orders differ")
    with np.load(art("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz"), allow_pickle=False) as z:
        support = z["row_support"]
    binding = read(art("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json"))["overlap"]["pairs"]
    archive = read(art("v122-mcs85-archive-records-verified", "E124", "mcs85_archive_records.json"))
    result = {"schema": "starlink-cross32-counter-replay/v1", "held_pair_count": len(pairs), "scores": score(pairs),
              "chosen": decode_record["chosen"], "pairs": pairs,
              "carries": [{**pair, "whole_high_byte_mask": masked_carry(prior, frames, support, pair, pn, decode_record["chosen"]),
                           "archive_binding": archive_binding(pair, binding, archive)} for pair in candidates],
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in receipts.items()}}
    path = experiment_output("E125")
    path.mkdir(parents=True, exist_ok=True)
    (path / "cross32_decoded_counters.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                       encoding="utf-8")
    print("Held same-tag pairs", len(pairs), result["scores"])
    print("Candidate carries and full-byte ablations", len(candidates))


if __name__ == "__main__":
    main()
