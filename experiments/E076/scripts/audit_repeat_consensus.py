"""Measure hard-decision repeat fusion on positions unseen by the registration test."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_to_code21_edge import recover_systematic_erasures
from audit_pn_padding_starts import CacheOnly, demap_exact, lattice_model
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mixed_slot import TYPES
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


REGISTRATION = 256
CONTROL_SHIFT = 128
CODE5_SEED = 2026112176


def fuse_two(values, masks):
    both = masks.all(axis=0)
    conflict = both & (values[0] != values[1])
    single = masks[0] ^ masks[1]
    predicted = np.where(masks[0], values[0], values[1])
    return predicted, both & ~conflict, single, conflict


def compare_section(values, masks, held, valid):
    predicted, agreed, single, conflict = fuse_two(values, masks)

    def score(chosen, guess=predicted):
        observed = chosen & valid
        return [int(observed.sum()), int(np.count_nonzero(observed & (guess != held)))]

    return {
        "reference_observed": int(valid.sum()),
        "agreed_both": score(agreed),
        "single_donor": score(single),
        "conflicts": int(np.count_nonzero(conflict & valid)),
        "target_only": int(np.count_nonzero(valid & ~masks.any(axis=0))),
        "individual_donors": [score(mask, value) for mask, value in zip(masks, values, strict=True)],
        "agreed_wrong_position": score(np.roll(agreed, CONTROL_SHIFT), np.roll(predicted, CONTROL_SHIFT)),
    }


def same_prefix(left, right, masks):
    known = masks.all(axis=0)
    support = int(known.sum())
    differences = int(np.count_nonzero((left != right) & known))
    return support >= .9 * REGISTRATION and differences <= .02 * support, [support, differences]


def matching_triples(values, masks):
    prefix, prefix_masks = values[:, :REGISTRATION], masks[:, :REGISTRATION]
    good = {}
    for a, b in combinations(range(len(values)), 2):
        passes, evidence = same_prefix(prefix[a], prefix[b], prefix_masks[[a, b]])
        if passes:
            good[a, b] = evidence
    return [entry for entry in combinations(range(len(values)), 3)
            if all(pair in good for pair in combinations(entry, 2))], good


def decode_frame(samples, rotation, signs):
    normalized = remove_header_gain(samples * rotation, signs).reshape(rotation.shape)
    # Use only this frame to decide which slots have lost both magnitude bits.
    axis_outer = np.isclose(abs(normalized.real), 1 / np.sqrt(2), atol=2e-6, rtol=0)
    axis_outer &= np.isclose(abs(normalized.imag), 1 / np.sqrt(2), atol=2e-6, rtol=0)
    return normalized.ravel(), axis_outer.all(axis=1)


def extract_allocation(decoded, alloc, demapper, amplitude):
    start = alloc["start"]
    width = alloc["width_resources"]
    count = alloc["count"]
    stop = start + width * count
    slots = np.arange(start, stop) // len(NONPILOT)
    data, validity = [], []
    for flat, four in decoded:
        bits, seen = demap_exact(flat[start:stop], demapper)
        mask = seen[:, None] & ~(four[slots, None] & amplitude[None, :])
        data.append(bits.reshape(count, -1))
        validity.append(mask.reshape(count, -1))
    return np.array(data), np.array(validity)


def triplet_checks(frames, values, masks, raw_masks, stop, reference):
    data_regions = {"systematic_after_registration": slice(REGISTRATION, stop),
                    "candidate_parity": slice(stop, None)}
    pn_registration = slice(stop - 1536, stop - 1024)
    pn_test = slice(stop - 512, stop)
    pn_known = masks[:, pn_registration]
    pn_hits = (values[:, pn_registration] == reference[pn_registration])
    pn_counts = pn_known.sum(axis=1)
    pn_verified = bool(((pn_counts >= .90 * 512) & ((pn_known & pn_hits).sum(axis=1) >= .85 * pn_counts)).all())
    out = []
    for index in range(3):
        use = [i for i in range(3) if i != index]
        row = {"target_frame": frames[index]}
        for region, sel in data_regions.items():
            row[region] = compare_section(values[use, sel], masks[use, sel], values[index, sel], masks[index, sel])
            row[region]["without_magnitude_erasure"] = compare_section(
                values[use, sel], raw_masks[use, sel], values[index, sel], raw_masks[index, sel])
        if pn_verified:
            row["frozen_pn15_control"] = compare_section(values[use, pn_test], masks[use, pn_test],
                                                        reference[pn_test], np.ones(512, dtype=bool))
            row["frozen_pn15_control"]["individual_target"] = [
                int(masks[index, pn_test].sum()),
                int(np.count_nonzero(masks[index, pn_test] & (values[index, pn_test] != reference[pn_test]))),
            ]
        out.append(row)
    counts = masks.sum(axis=0)
    ones = (values & masks).sum(axis=0)
    no_conflict = (ones == 0) | (ones == counts)
    unanimous_covered = (counts >= 2) & no_conflict
    return out, unanimous_covered, (ones > 0)


def audit_code5_triplet(values, valid, matrix, training):
    majority = (values.sum(axis=0) >= 2).astype(np.uint8)
    support = valid.sum(axis=0)
    positive = (values & valid).sum(axis=0)
    unanimous = (support >= 2) & ((positive == 0) | (positive == support))
    consensus = (positive > 0).astype(np.uint8)
    k = matrix.shape[0]
    observed = unanimous[k:]
    unknown = ~unanimous[:k]
    known_system = consensus[:k].copy()
    known_system[unknown] = 0
    residual = ((known_system @ matrix) & 1) ^ consensus[k:]

    def parity_errors(bits):
        return np.count_nonzero(((bits[..., :k] @ matrix) & 1) ^ bits[..., k:], axis=-1)

    if unknown.any():
        held = recover_systematic_erasures(matrix[unknown], residual, observed, training)
    else:
        held = {"validated": not residual[observed].any(), "systematic_guess": [],
                "folds": [], "unobserved_parity_count": int((~observed).sum())}
    rebuilt = None
    if held["validated"]:
        known_system[unknown] = held.pop("systematic_guess")
        rebuilt = np.r_[known_system, (known_system @ matrix) & 1]
        if np.any((rebuilt != consensus) & unanimous):
            raise ValueError("Erasure reconstruction does not preserve unanimous observations")
    return {"raw_parity_errors": parity_errors(values).tolist(),
            "majority_parity_errors": int(parity_errors(majority)),
            "unanimous_system_count": int((~unknown).sum()),
            "unanimous_parity_count": int(observed.sum()), "erasure_validation": held}, majority, rebuilt


def code5_group(frames, header, decoded, matrix, training, archive, use_partial):
    entries = header["h1"]["layout20"]["entries"]
    if len(entries) != 1 or entries[0]["mcs_candidate"] != 5:
        return []
    size = entries[0]["quantity"]
    start = 228
    words = []
    if header["h2_start"] != start + size * 2048:
        raise ValueError("Code 5 allocation differs from frozen H2 start")
    symbols = np.array([flat[start:header["h2_start"]].reshape(size, 2048) for flat, _ in decoded])
    bpsk = np.isclose(abs(symbols.real), 1, atol=2e-6, rtol=0) & (abs(symbols.imag) <= 2e-6)
    for index in range(size):
        bits = (symbols[:, index].real < 0).astype(np.uint8)
        valid = bpsk[:, index]
        triples, _ = matching_triples(bits, valid)
        for combo in triples:
            selected = list(combo)
            if (valid[selected].mean(axis=1) < (.90 if use_partial else 1)).any():
                continue
            subset = [frames[i] for i in selected]
            result, majority, rebuilt = audit_code5_triplet(bits[selected], valid[selected], matrix, training)
            key = f"code5_{subset[0]}_{subset[1]}_{subset[2]}_{index}"
            archive[f"{key}_majority"] = np.packbits(majority)
            archive[f"{key}_hard"] = np.packbits(bits[selected], axis=-1)
            if use_partial:
                archive[f"{key}_seen_bpsk"] = np.packbits(valid[selected], axis=-1)
            if rebuilt is not None:
                archive[f"{key}_reconstructed"] = np.packbits(rebuilt)
            words.append({"frames": subset, "word": index, **result})
    return words


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--code5-control"], ["--code5-control-partial"]):
        raise ValueError("Supported options: --code5-control and --code5-control-partial")
    followup = bool(sys.argv[1:])
    use_partial = "--code5-control-partial" in sys.argv[1:]
    inputs = {}
    headers = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))["frames"]
    pad = read(checked_artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json", inputs))
    label_path = checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs)
    labels32 = read(label_path)
    with np.load(checked_artifact("v026-ut-qam-extract-repeat", "E028", "ut_qam_symbols.npz", inputs),
                 allow_pickle=False) as data:
        points = data["constellation_points"]
        labels = data["frozen_gray_mapping"]
    amplitude = np.ptp(labels[np.isclose(abs(points), abs(points).max())], axis=0) == 0
    points32 = np.array([complex(row["i"], row["q"]) for row in labels32["cross32_mapping"]])
    mapping32 = labels_as_bits(np.array([row["label"] for row in labels32["cross32_mapping"]]), 5)
    models = {"16QAM": (lattice_model(points, labels), amplitude),
              "cross32": (lattice_model(points32, mapping32), np.zeros(5, dtype=bool))}
    frozen_pn = equality_predictions(np.arange(16385), pn_bits(), labels32["frozen_pn_model"])
    by_header = defaultdict(list)
    for item in headers:
        key = tuple(tuple(item[name]["words_hex"]) for name in ("h1", "h2"))
        by_header[key].append(item["frame"])
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    all_groups = []
    bits_and_masks = {}
    if followup:
        source5 = checked_artifact("v010-code5-parity-02", "E012", "code5_observations.npz", inputs)
        with np.load(source5, allow_pickle=False) as archive:
            matrix = archive["parity_generator"]
        group5 = sorted(np.random.default_rng(CODE5_SEED).permutation(8)[:4].tolist())
        mask5 = np.isin(np.arange(1024) // 128, group5)
        training5 = [mask5, ~mask5]
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        source = archive["yDataDec"]
        for frames in (group for group in by_header.values() if len(group) >= 3):
            rejected = 0
            words = []
            decoded = [decode_frame(as_complex(source[frame, 1:, :])[..., NONPILOT % 1024], rotation, signs)
                       for frame in frames]
            for alloc in pad["word_windows_and_onsets"][frames[0]]["allocations"]:
                label = alloc["label"]
                edge = pad["marker_systematic_ends"][str(label)]
                model, removed = models[TYPES[label]]
                data, valid = extract_allocation(decoded, alloc, model, removed)
                for word in range(alloc["count"]):
                    values, masks = data[:, word], valid[:, word]
                    triples, good = matching_triples(values, masks)
                    rejected += int(not triples)
                    if not triples:
                        continue
                    start = alloc["start"] + word * alloc["width_resources"]
                    span_slots = (np.arange(start, start + alloc["width_resources"]) // len(NONPILOT))
                    for indices in triples:
                        selected = list(indices)
                        run_frames = [frames[i] for i in selected]
                        local_masks = masks[selected]
                        original = local_masks.copy().reshape(3, -1, len(removed))
                        # Promote only the mask's magnitude lanes back for the raw comparison.
                        visible = original[..., ~removed].any(axis=-1)
                        original = np.broadcast_to(visible[..., None], original.shape).reshape(local_masks.shape)
                        result, trust, consensus = triplet_checks(
                            run_frames, values[selected], local_masks, original, edge, frozen_pn)
                        key = f"{run_frames[0]}_{run_frames[1]}_{run_frames[2]}_{start}"
                        bits_and_masks[f"{key}_bits"] = np.packbits(consensus & trust)
                        bits_and_masks[f"{key}_mask"] = np.packbits(trust)
                        words.append({
                            "frames": run_frames, "label": label, "resource": start, "word": word,
                            "registration_pairs": [good[pair] for pair in combinations(indices, 2)],
                            "four_corner_slots": [(np.unique(span_slots[decoded[i][1][span_slots]]) + 2).tolist()
                                                  for i in selected],
                            "unopposed_with_two_support": int(trust.sum()), "total_bits": len(trust),
                            "region_stop": edge, "leave_one_out": result,
                        })
            result_group = {"same_headers": frames, "word_positions_without_triplet": rejected, "triplets": words}
            if followup:
                result_group["code5_control"] = code5_group(
                    frames, headers[frames[0]], decoded, matrix, training5, bits_and_masks, use_partial)
            all_groups.append(result_group)
            print("same-header group", frames[0], len(frames), "word triplets", len(words), flush=True)
    prior = {row["bytes"][0]: row["sha256"] for row in pad["source"]["pages"]}
    if any(prior[start] != row["sha256"] for start, row in reader.receipts.items()):
        raise ValueError("Archive bytes differ from E070")
    result = {
        "schema": "starlink-repeat-consensus/v1", "registration_bits": REGISTRATION,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                   "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
        "groups": all_groups,
    }
    if followup:
        result["code5_control_block_split"] = {"seed": CODE5_SEED, "first_fold_calibration_blocks": group5}
    if use_partial:
        result["minimum_code5_visible_fraction"] = .90
    out = experiment_output("E076")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "repeat_consensus.npz", **bits_and_masks)
    (out / "repeat_consensus.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
