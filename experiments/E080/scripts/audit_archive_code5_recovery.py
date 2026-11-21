"""Reprocess the Code 5 archive with independent axis and edge controls."""

from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_byte_records import (BIT_LENGTH, START_RESOURCE, add_evidence, bpsk_mask, erasure_word, structure)
from audit_code5_frame_count import PHASE_LIMIT, RATE, counter_at, descramble, off_grid_links
from audit_code5_to_code21_edge import CALIBRATION_CARRIERS, carrier_axis
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


def normalize_axis(values):
    squared = values[:1024] ** 2
    axis = np.sqrt(complex(np.median(squared.real), np.median(squared.imag)))
    if not np.isfinite(axis) or abs(axis) < .5 or axis.real <= 0:
        return None
    return values / axis


def recover_without_timing(values, matrix):
    carriers = carrier_axis(START_RESOURCE, BIT_LENGTH)
    normalized = normalize_axis(values)
    candidates = {
        "header_gain": (values, None),
        "header_gain_erase_511": (values, 511),
        "word_axis": (normalized, None),
        "word_axis_erase_511": (normalized, 511),
        "word_axis_erase_minus_512": (normalized, -512),
    }
    receipts = {}
    accepted = None
    for name, (source, carrier) in candidates.items():
        if source is None:
            receipts[name] = None
            continue
        used = source.copy()
        if carrier is not None:
            used[carriers == carrier] = 0
        output = erasure_word(used, matrix)
        receipts[name] = None if output is None else output[1]
        if output is not None and name == "word_axis_erase_511":
            accepted = output[0], used
    return accepted, receipts


def checked_single_error(bits, checks, first):
    syndrome = (checks @ bits) & 1
    if not syndrome.any():
        return None
    guesses = [np.flatnonzero(np.all(checks[fold] == syndrome[fold, None], axis=0))
               for fold in (first, ~first)]
    if len(guesses[0]) == len(guesses[1]) == 1 and guesses[0][0] == guesses[1][0]:
        return int(guesses[0][0])
    return None


def remaining_word(values, matrix, checks, first):
    normalized = normalize_axis(values)
    if normalized is None:
        return None, {"method": None, "reason": "no stable word axis"}
    carriers = carrier_axis(START_RESOURCE, BIT_LENGTH)
    erased = np.where(np.isin(carriers, CALIBRATION_CARRIERS), 0, normalized)
    result = erasure_word(erased, matrix)
    if result is not None:
        return (result[0], erased), {"method": "six_edge_carriers", **result[1]}
    observed = bpsk_mask(normalized)
    if not observed.all():
        return None, {"method": None, "unobserved_positions": np.flatnonzero(~observed).tolist()}
    bits = (normalized.real < 0).astype(np.uint8)
    bit = checked_single_error(bits, checks, first)
    if bit is None:
        return None, {"method": None, "reason": "no unique common single-error candidate"}
    original = int(bits[bit])
    bits[bit] ^= 1
    assert not np.any((checks @ bits) & 1)
    normalized[bit] = 0
    return (bits, normalized), {"method": "single_checked_flip", "position": bit, "original": original,
                                 "carrier": int(carriers[bit]), "replacement": int(bits[bit])}


def archive_run(headers, matrix, pn, previous_words, complete):
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 5)) for k in NONPILOT]).conj()
    evidence, receipts = {}, []
    checks = np.column_stack([matrix.T, np.eye(1024, dtype=np.uint8)])
    first = np.isin(np.arange(1024) // 128, [0, 2, 3, 6])
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        source = archive["yDataDec"]
        if source.shape != (1009, 301, 1024):
            raise ValueError("Unexpected Qin archive size")
        for start in range(0, len(headers), source.chunks[0]):
            stop = min(start + source.chunks[0], len(headers))
            grids = as_complex(source[start:stop, 1:4, :])[..., NONPILOT % 1024]
            for index, grid in enumerate(grids, start):
                z = remove_header_gain(grid * rotation, signs)[START_RESOURCE:START_RESOURCE + BIT_LENGTH]
                accepted, arms = recover_without_timing(z, matrix)
                established = index in previous_words
                entry = {"archive_index": index, "e078_verified": established, "arms": arms, "counter": None}
                receipts.append(entry)
                if accepted is None and complete:
                    accepted, entry["fallback"] = remaining_word(z, matrix, checks, first)
                if accepted is None:
                    continue
                bits, used = accepted
                data = descramble(bits, pn)
                info = structure(data)
                entry.update(pointer=data[0], internal_marker=info["anchor_match"], counter=counter_at(data),
                             observed_bits=int(bpsk_mask(used).sum()))
                if established:
                    entry["e078_word_same"] = bool(np.array_equal(np.packbits(bits), previous_words[index]))
                else:
                    add_evidence(evidence, index, 0, bits, used)
    pages = [reader.receipts[key] for key in sorted(reader.receipts)]
    return receipts, evidence, pages


def attach_timing(rows, headers, split, reference):
    phase = reference["archive_phase_cycles"]
    offset = reference["results"]["qin_archive"]["scores"]["dynamic_positive"]["offset"]
    calibration = set(split["calibration_groups"])
    groups = split["archive_chunk_group_ids"]
    summary = Counter()
    for row, header in zip(rows, headers, strict=True):
        index = row["archive_index"]
        group = groups[index // 13]
        on_calibration = group in calibration
        cycles = RATE * header["toa_s"] - phase
        nearest = round(cycles)
        residual = cycles - nearest
        row.update(group=group, calibration=on_calibration, waveform_frame=nearest,
                   waveform_residual_cycles=residual, on_waveform_grid=abs(residual) <= PHASE_LIMIT)
        if row["counter"] is not None:
            row["counter_minus_prediction"] = row["counter"] - nearest - offset
        partition = "calibration" if on_calibration else "held"
        if row["e078_verified"]:
            if row.get("e078_word_same") is not True:
                summary[f"{partition}_previous_not_reproduced"] += 1
            continue
        summary[f"{partition}_previously_unresolved"] += 1
        for arm, details in row["arms"].items():
            if details is not None:
                summary[f"{partition}_{arm}"] += 1
        if row["counter"] is None:
            continue
        if row["internal_marker"]:
            summary[f"{partition}_new_marker"] += 1
        if row["on_waveform_grid"]:
            summary[f"{partition}_new_on_grid"] += 1
            if row["counter_minus_prediction"] == 0:
                summary[f"{partition}_new_matching_counter"] += 1
        else:
            summary[f"{partition}_new_off_grid"] += 1
    return dict(sorted(summary.items()))


def compare_linked_words(links, words):
    comparable = []
    for link in links["links"]:
        source = link["archive_index"]
        for destination in link["matches"]["0"]:
            target = destination["archive_index"]
            bits1, mask1 = words[source]
            bits2, mask2 = words[target]
            both = mask1 & mask2
            differences = bits1 ^ bits2
            comparable.append({"source": source, "destination": target,
                               "differing_bits": int(np.unpackbits(differences).sum()),
                               "jointly_observed_bits": int(np.unpackbits(both).sum()),
                               "jointly_observed_differences": int(np.unpackbits(differences & both).sum())})
    return comparable


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--complete-and-link"]):
        raise ValueError("Only --complete-and-link is supported")
    complete = bool(sys.argv[1:])
    inputs = {}
    def artifact(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    headers = read(artifact("v034-qin-header-survey", "E036", "qin_header_survey.json"))["frames"]
    split = read(artifact("v075-dense-padding-diagnostic", "E077", "dense_padding_folds.json"))
    reference = read(artifact("v077-code5-time-links", "E079", "code5_frame_count.json"))
    model = read(artifact("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    with np.load(artifact("v010-code5-parity-02", "E012", "code5_observations.npz"), allow_pickle=False) as z:
        matrix = z["parity_generator"]
    with np.load(artifact("v076-code5-byte-words", "E078", "archived_code5_words.npz"), allow_pickle=False) as z:
        previous = {int(key.split("_")[1]): z[key] for key in z.files if key.endswith("_word_0_bits")}
        previous_masks = {i: z[f"frame_{i}_word_0_observed"] for i in previous}
    rows, evidence, pages = archive_run(headers, matrix, pn, previous, complete)
    summary = attach_timing(rows, headers, split, reference)
    report = {"schema": "starlink-archive-code5-recovery/v1", "summary": summary, "frames": rows,
              "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE, "pages": pages},
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()}}
    if complete:
        repeats = read(artifact("v072-padding-parity-final", "E074", "padding_parity_transfer.json"))
        timing = [(r["waveform_frame"], r["waveform_residual_cycles"], r["on_waveform_grid"]) for r in rows]
        comparable = [{**r, "record_index": r["archive_index"]} for r in rows if r["counter"] is not None]
        offset = reference["results"]["qin_archive"]["scores"]["dynamic_positive"]["offset"]
        pairs = repeats["nearby_diagnostic"]["same_header_pairs"]
        report["off_grid_links"] = off_grid_links(comparable, headers, timing, offset, pairs)
        words = {i: (bits, previous_masks[i]) for i, bits in previous.items()}
        for r in rows:
            i = r["archive_index"]
            if i not in words and r["counter"] is not None:
                words[i] = (evidence[f"frame_{i}_word_0_bits"], evidence[f"frame_{i}_word_0_observed"])
        report["linked_codewords"] = compare_linked_words(report["off_grid_links"], words)
        for r in comparable:
            if r["on_waveform_grid"] and r["counter_minus_prediction"]:
                r["on_waveform_grid"] = False
        report["all_timing_exceptions"] = off_grid_links(comparable, headers, timing, offset, pairs)
        report["inputs_sha256"] = {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()}
    out = experiment_output("E080")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "recovered_code5_words.npz", **evidence)
    (out / "archive_code5_recovery.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
