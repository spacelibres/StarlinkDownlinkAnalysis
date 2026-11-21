"""Compare an unaligned Code 5 field with frame epochs measured from the waveform."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


RELATIVE_BIT = 421
RATE = 750
MODULUS = 1 << 32
PHASE_LIMIT = .01


def counter_at(word, delta=0, absolute=None, width=32):
    start = 8 * word[0] + RELATIVE_BIT if absolute is None else absolute
    start += delta
    return (int.from_bytes(word, "little") >> start) & ((1 << width) - 1)


def fit_offset(values, frame_numbers, modulus=MODULUS):
    counts = Counter((int(value) - int(n)) % modulus for value, n in zip(values, frame_numbers, strict=True))
    return counts.most_common(1)[0][0]


def wrap_phase(value):
    return (value + .5) % 1 - .5


def measure_timing(cycles, calibration):
    waveform_phase = cycles[calibration] % 1
    grid = np.arange(1000) / 1000
    seed = grid[np.count_nonzero(abs(wrap_phase(waveform_phase[:, None] - grid)) <= PHASE_LIMIT, axis=0).argmax()]
    deltas = wrap_phase(waveform_phase - seed)
    phase = float((seed + np.median(deltas[abs(deltas) <= PHASE_LIMIT])) % 1)
    n = np.rint(cycles - phase).astype(int)
    residual = cycles - phase - n
    return phase, [(int(i), float(r), bool(abs(r) <= PHASE_LIMIT)) for i, r in zip(n, residual, strict=True)]


def descramble(bits, pn):
    return np.packbits(bits[:len(pn)] ^ pn, bitorder="little").tobytes()


def features(data, layout):
    ids = layout["record_ids"] if layout["full_match"] or layout["additional_layouts"]["reverse_tag_variant"] else []
    return {
        "counter": counter_at(data),
        "nearby": {str(delta): counter_at(data, delta=delta) for delta in (-2, -1, 1, 2)},
        "fixed_bit": {str(position): counter_at(data, absolute=position) for position in (613, 677)},
        "offset_bytes": data[0],
        "header_marker": layout["anchor"],
        "preceding_bit": counter_at(data, delta=-1, width=1),
        "following_bits": counter_at(data, delta=32, width=4),
        "record_values": ids,
        "shared_upper_13": ids[0] >> 3 if ids else None,
        "short_record_selectors": [v & 7 for v in ids],
    }


def first_words(report, bits):
    return {
        row.get("frame", row.get("candidate_index")): word
        for row, word in zip(report["words"], bits, strict=True)
        if row["word"] == 0 and row["start_resource"] == 228
    }


def score(rows, mode):
    valid = [row for row in rows if row["on_waveform_grid"]]
    arms = {
        "dynamic_positive": ([row["counter"] for row in valid], [row["waveform_frame"] for row in valid], MODULUS),
        "dynamic_negative": ([row["counter"] for row in valid], [-row["waveform_frame"] for row in valid], MODULUS),
        "archive_sequence": ([row["counter"] for row in valid], [row["record_index"] for row in valid], MODULUS),
        **{f"bit_shift_{d}": ([row["nearby"][str(d)] for row in valid],
                             [row["waveform_frame"] for row in valid], MODULUS) for d in (-2, -1, 1, 2)},
        **{f"absolute_bit_{d}": ([row["fixed_bit"][str(d)] for row in valid],
                                [row["waveform_frame"] for row in valid], MODULUS) for d in (613, 677)},
    }
    short = [row for row in valid if row["shared_upper_13"] is not None]
    controls = {}
    for name, (values, frames, modulus) in arms.items():
        controls[name] = score_arm(valid, values, frames, modulus)
    if short:
        for name, f in (("short_record_plus_one", 1), ("short_record_minus_one", -1), ("short_record_constant", 0)):
            controls[name] = score_arm(short, [row["shared_upper_13"] for row in short],
                                       [f * row["waveform_frame"] for row in short], 1 << 13)
    main = controls["dynamic_positive"]
    off_grid = [row for row in rows if not row["on_waveform_grid"]]
    return {
        "mode": mode, "scores": controls, "waveform_grid_rows": len(valid),
        "distinct_grid_counters": len({row["counter"] for row in valid}),
        "grid_counter_range": [min(row["counter"] for row in valid), max(row["counter"] for row in valid)],
        "or_variation_mask": int(np.bitwise_or.reduce([row["counter"] ^ valid[0]["counter"] for row in valid])),
        "off_grid": [{"record_index": row["record_index"], "residual_cycles": row["waveform_residual_cycles"],
                      "counter": row["counter"],
                      "difference_from_main_grid_count": int(row["counter"] - (
                          (row["waveform_frame"] + main["offset"]) % MODULUS))} for row in off_grid],
    }


def score_arm(rows, values, frames, modulus):
    known = [i for i, row in enumerate(rows) if row["calibration"]]
    held = [i for i, row in enumerate(rows) if not row["calibration"]]
    if not known:
        return {"offset": None, "held": {"supported": len(held), "matches": None}}
    offset = fit_offset([values[i] for i in known], [frames[i] for i in known], modulus)

    def assess(index):
        return {"supported": len(index),
                "matches": sum(values[i] == (frames[i] + offset) % modulus for i in index)}

    return {"offset": offset, "calibration": assess(known), "held": assess(held)}


def build_raw(rows, words, waveform, pn):
    result = []
    for row in rows:
        if row["result"] is None:
            continue
        index = row["frame"]
        n, r = waveform[index]
        result.append({"record_index": index, "waveform_frame": n, "waveform_residual_cycles": r,
                       "on_waveform_grid": True, "calibration": len(result) == 0,
                       **features(descramble(words[index], pn), row["result"])})
    return result


def off_grid_links(rows, headers, waveform, offset, repeat_rows):
    timing = {}
    for index, (number, _, admitted) in enumerate(waveform):
        if admitted:
            timing.setdefault(number, []).append(index)
    repeated = {tuple(pair["frames"]): len(pair["repeated_words"]) for pair in repeat_rows}
    counters = {row["record_index"]: row["counter"] for row in rows}
    links = []
    for row in rows:
        if row["on_waveform_grid"]:
            continue
        index = row["record_index"]
        predicted = row["counter"] - offset
        matches = {}
        for delta in range(-16, 17):
            candidates = []
            for other in timing.get(predicted + delta, []):
                same = all(headers[index][key]["words_hex"] == headers[other][key]["words_hex"]
                           for key in ("h1", "h2"))
                candidates.append({
                    "archive_index": other, "shared_h1_h2": same,
                    "time_difference_s": headers[other]["toa_s"] - headers[index]["toa_s"],
                    "candidate_counter_if_observed": counters.get(other),
                    "previous_repeated_words": repeated.get(tuple(sorted((index, other))), None),
                })
            matches[str(delta)] = candidates
        links.append({"archive_index": index, "predicted_waveform_frame": predicted, "matches": matches})
    totals = {str(delta): sum(any(candidate["shared_h1_h2"] for candidate in row["matches"][str(delta)])
                              for row in links) for delta in range(-16, 17)}
    return {"both_headers_by_frame_shift": totals, "links": links}


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--off-grid-links"]):
        raise ValueError("Only --off-grid-links is supported")
    include_links = bool(sys.argv[1:])
    inputs = {}
    plan = "v076-code5-byte-words"
    sources = read(checked_artifact(plan, "E078", "code5_byte_records.json", inputs))["raw"]
    binary = checked_artifact(plan, "E078", "archived_code5_words.npz", inputs)
    split = read(checked_artifact("v075-dense-padding-diagnostic", "E077", "dense_padding_folds.json", inputs))
    headers = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))["frames"]
    model = read(checked_artifact("v071-pn-parity-order", "E073", "pn_parity_order.json", inputs))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    chunks = split["archive_chunk_group_ids"]
    groups = np.array([chunks[row["frame"] // 13] for row in headers])
    calibrated = np.isin(groups, split["calibration_groups"])
    phase, waveform = measure_timing(RATE * np.array([row["toa_s"] for row in headers]), calibrated)
    archived = []
    with np.load(binary, allow_pickle=False) as arrays:
        for row in sources["qin_archive"]:
            layout = row["result"] or row.get("erasure_result", {}).get("result")
            if layout is None:
                continue
            index = row["frame"]
            n, r, admitted = waveform[index]
            bits = np.unpackbits(arrays[f"frame_{index}_word_0_bits"])
            archived.append({"record_index": index, "waveform_frame": n, "waveform_residual_cycles": r,
                             "on_waveform_grid": admitted, "calibration": bool(calibrated[index]),
                             "waveform_group": int(groups[index]), "from_erasures": "erasure_result" in row,
                             **features(descramble(bits, pn), layout)})

    old, ut = "v010-code5-parity-02", "v022-ut-replay"
    old_info = read(checked_artifact(old, "E012", "code5_parity.json", inputs))
    ut_info = read(checked_artifact(ut, "E014", "ut_transfer.json", inputs))
    with np.load(checked_artifact(old, "E012", "code5_observations.npz", inputs), allow_pickle=False) as z:
        first_old = first_words(old_info, z["hard_bits"])
    with np.load(checked_artifact(ut, "E014", "ut_observations.npz", inputs), allow_pickle=False) as z:
        first_ut = first_words(ut_info, z["hard_code5_words"])
    raw_toas = {row["candidate_index"]: row["pss"]["epoch"] for row in ut_info["candidates"] if "pss" in row}
    intervals = {i: (t - raw_toas[min(first_ut)]) / 640000 for i, t in raw_toas.items() if i in first_ut}
    time_ut = {i: (round(t), t - round(t)) for i, t in intervals.items()}
    if max(abs(v[1]) for v in time_ut.values()) > PHASE_LIMIT:
        raise ValueError("UT independently acquired epochs no longer share the assumed grid")
    old_rows = build_raw(sources["legacy_480"], first_old, {i: (i, None) for i in first_old}, pn)
    ut_rows = build_raw(sources["ut_250"], first_ut, time_ut, pn)
    cohorts = {"legacy_480": old_rows, "ut_250": ut_rows, "qin_archive": archived}
    modes = {"legacy_480": "per-capture origin from first resolved word; previously measured frame index",
             "ut_250": "per-capture origin from first resolved word; PSS epoch increments",
             "qin_archive": "E077 connected group split; TOA phase learned from training observations only"}
    result = {
        "schema": "starlink-code5-frame-count/v1", "relative_bit_offset": RELATIVE_BIT,
        "rate_hz": RATE, "archive_phase_cycles": phase,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "results": {name: score(rows, modes[name]) for name, rows in cohorts.items()},
        "short_record_totals": {
            name: {"words": sum(bool(row["record_values"]) for row in rows),
                   "bits_1_and_2_zero": sum(bool(row["record_values"]) and
                                                    all(not (v & 6) for v in row["record_values"]) for row in rows),
                   "distinct_upper_13": len({row["shared_upper_13"] for row in rows
                                            if row["shared_upper_13"] is not None})}
            for name, rows in cohorts.items()
        },
        "rows": cohorts,
    }
    if include_links:
        repeats_path = checked_artifact("v072-padding-parity-final", "E074", "padding_parity_transfer.json", inputs)
        repeats = read(repeats_path)["nearby_diagnostic"]["same_header_pairs"]
        offset = result["results"]["qin_archive"]["scores"]["dynamic_positive"]["offset"]
        result["off_grid_links"] = off_grid_links(archived, headers, waveform, offset, repeats)
        result["inputs_sha256"] = {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()}
    out = experiment_output("E079")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_frame_count.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, row in result["results"].items():
        print(name, row["scores"]["dynamic_positive"], "off grid", len(row["off_grid"]))


if __name__ == "__main__":
    main()
