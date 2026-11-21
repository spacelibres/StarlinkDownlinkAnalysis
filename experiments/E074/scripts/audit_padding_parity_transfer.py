"""Try short-prefix prediction of high-order parity with observed held coordinates."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_pn_padding_starts import CacheOnly, demap_exact, lattice_model
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mixed_slot import TYPES
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import as_complex


PREFIX = 1024
SEED = 2026112174
PERMUTATIONS = 199
FRAME_GUARD = 3


def pack_key(prefix):
    return int.from_bytes(np.packbits(prefix).tobytes(), "big") * 2 + 1


def affine_basis(keys):
    basis = {}
    for index, value in enumerate(keys):
        used = 1 << index
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value, used
                break
            column, coefficient = basis[pivot]
            value ^= column
            used ^= coefficient
    return basis


def representation(value, basis):
    used = 0
    while value:
        pivot = value.bit_length() - 1
        if pivot not in basis:
            return None
        column, coefficient = basis[pivot]
        value ^= column
        used ^= coefficient
    return used


def projected_words(inputs, followup):
    padding = read(checked_artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json", inputs))
    cross = read(checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs))
    with np.load(checked_artifact("v026-ut-qam-extract-repeat", "E028", "ut_qam_symbols.npz", inputs),
                 allow_pickle=False) as archive:
        model16 = lattice_model(archive["constellation_points"], archive["frozen_gray_mapping"])
    points = np.array([complex(row["i"], row["q"]) for row in cross["cross32_mapping"]])
    labels = labels_as_bits(np.array([row["label"] for row in cross["cross32_mapping"]]), 5)
    demappers = {"16QAM": model16, "cross32": lattice_model(points, labels)}
    stops = padding["marker_systematic_ends"]
    selected = defaultdict(list)
    all_frames = padding["word_windows_and_onsets"]
    for frame in all_frames:
        for allocation in frame["allocations"]:
            for word in allocation["strong_words"]:
                if word["last"] and word["two_nat_minmax"][1] <= PREFIX:
                    selected[frame["frame"]].append((frame, allocation, word))
    slots = np.arange(2, 302)
    pn_rotation = np.column_stack([pn_reference(int(k), slots) for k in NONPILOT]).conj()
    reference_bits = encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48]
    reference_signs = 1 - 2 * reference_bits.astype(float)
    pn = equality_predictions(np.arange(16385), pn_bits(), cross["frozen_pn_model"])
    results = defaultdict(list)
    repeated = []
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as file:
        source = file["yDataDec"]
        if source.shape != (1009, 301, 1024):
            raise ValueError("Unexpected Qin archive geometry")
        def normalized_frame(frame_index):
            grid = as_complex(source[frame_index, 1:, :])[..., NONPILOT % 1024]
            return remove_header_gain(grid * pn_rotation, reference_signs)

        for frame_index, requests in sorted(selected.items()):
            flat = normalized_frame(frame_index)
            for frame, alloc, word in requests:
                label = alloc["label"]
                start = word["resource_start"]
                width = alloc["width_resources"]
                hard, visible = demap_exact(flat[start:start + width], demappers[TYPES[label]])
                bits, mask = hard.ravel(), np.repeat(visible, hard.shape[-1])
                edge = stops[str(label)]
                bound = (word["two_nat_minmax"][1] + 7) // 8 * 8
                matches = int(np.count_nonzero(mask[bound:edge] & (bits[bound:edge] == pn[bound:edge])))
                prefix, valid = bits[:PREFIX].copy(), mask[:PREFIX].copy()
                adjusted, adjusted_valid = prefix.copy(), valid.copy()
                adjusted[bound:] = pn[bound:PREFIX]
                adjusted_valid[bound:] = True
                results[label].append({
                    "frame": frame_index, "group": frame["group"], "held": frame["held"],
                    "start_resource": start, "bound": bound, "edge": edge,
                    "tail_observed": int(mask[bound:edge].sum()),
                    "tail_disagreements": int(mask[bound:edge].sum()) - matches,
                    "prefix": prefix, "prefix_mask": valid,
                    "conditional_prefix": adjusted, "conditional_prefix_mask": adjusted_valid,
                    "parity": bits[edge:], "parity_mask": mask[edge:],
                })
        if followup:
            header_path = checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs)
            headers = read(header_path)["frames"]
            repeated = nearby_frames(headers, all_frames, normalized_frame, demappers, stops)
    old_pages = {row["bytes"][0]: row["sha256"] for row in padding["source"]["pages"]}
    if any(old_pages[start] != row["sha256"] for start, row in reader.receipts.items()):
        raise ValueError("Source archive bytes differ from the E070 page receipts")
    return results, [reader.receipts[page] for page in sorted(reader.receipts)], repeated


def nearby_frames(headers, frames, normalized_frame, demappers, stops):
    def same_headers(a, b):
        return all(a[name]["words_hex"] == b[name]["words_hex"] for name in ("h1", "h2"))

    pairs = []
    for lag in range(1, FRAME_GUARD + 1):
        for left, right in zip(headers, headers[lag:]):
            if not same_headers(left, right):
                continue
            a, b = left["frame"], right["frame"]
            corners = [normalized_frame(index) for index in (a, b)]
            found, count = [], 0
            for alloc in frames[a]["allocations"]:
                label = alloc["label"]
                width = alloc["width_resources"]
                edge = stops[str(label)]
                alphabet = demappers[TYPES[label]]
                start = alloc["start"]
                size = alloc["count"]
                decoded = [demap_exact(flat[start:start + size * width], alphabet) for flat in corners]
                (v, vmask), (w, wmask) = decoded
                valid_symbols = vmask & wmask
                valid = np.repeat(valid_symbols, v.shape[-1]).reshape(size, -1)
                mismatch = (v != w).reshape(valid.shape) & valid
                found_bits = np.stack([valid[:, :256].sum(axis=1), valid[:, edge:].sum(axis=1)]).T
                errors = np.stack([mismatch[:, :256].sum(axis=1), mismatch[:, edge:].sum(axis=1)]).T
                compared = (found_bits >= .90 * np.array([256, valid.shape[1] - edge])).all(axis=1)
                matched = compared & (errors <= .02 * found_bits).all(axis=1)
                count += int(compared.sum())
                for index in np.flatnonzero(matched):
                    found.append({"label": label, "word": int(index), "resource": start + int(index) * width,
                                  "prefix_and_parity_known": found_bits[index].tolist(),
                                  "prefix_and_parity_disagreements": errors[index].tolist(),
                                  "systematic_known": int(valid[index, :edge].sum()),
                                  "systematic_disagreements": int(mismatch[index, :edge].sum())})
            pairs.append({"frames": [a, b], "groups": [frames[a]["group"], frames[b]["group"]],
                          "crosses_split": frames[a]["held"] != frames[b]["held"],
                          "toa_interval_s": right["toa_s"] - left["toa_s"],
                          "assessed_words": count, "repeated_words": found})
    return pairs


def nearest_predictions(cal, held, prefix, permutations, guard=0):
    keys = np.array([row[prefix] for row in cal])
    masks = np.array([row[prefix + "_mask"] for row in cal])
    parity = np.array([row["parity"] for row in cal])
    known_parity = np.array([row["parity_mask"] for row in cal])
    frames = np.array([row["frame"] for row in cal])
    receipts, scores = [], defaultdict(list)
    for target in held:
        known = masks & target[prefix + "_mask"]
        support = known.sum(axis=1)
        mismatch = np.count_nonzero(known & (keys != target[prefix]), axis=1)
        both = known_parity & target["parity_mask"]
        parity_support = both.sum(axis=1)
        parity_diff = np.count_nonzero(both & (parity != target["parity"]), axis=1)
        eligible = ((support >= .9 * PREFIX) & (parity_support >= .9 * len(target["parity"])) &
                    (abs(frames - target["frame"]) > guard))
        distance = np.divide(mismatch, support, out=np.full(len(cal), np.inf), where=eligible)
        winners = np.flatnonzero(distance == distance.min())
        item = {"frame": target["frame"], "start_resource": target["start_resource"],
                "eligible_donors": int(eligible.sum()), "nearest": None}
        receipts.append(item)
        if not eligible.any() or len(winners) != 1:
            continue
        best = int(winners[0])
        use = np.flatnonzero(eligible)
        # Preserve the same donor pool by taking the first eligible donor in each permutation.
        controls = permutations[np.arange(len(permutations)), eligible[permutations].argmax(axis=1)]
        parity_error = parity_diff[use] / parity_support[use]
        observed_rate = float(parity_diff[best] / parity_support[best])
        scores[target["group"]].append(np.r_[observed_rate, parity_diff[controls] / parity_support[controls]])
        item["nearest"] = {
            "frame": cal[best]["frame"], "start_resource": cal[best]["start_resource"],
            "prefix_intersection": int(support[best]), "prefix_disagreements": int(mismatch[best]),
            "parity_intersection": int(parity_support[best]), "parity_disagreements": int(parity_diff[best]),
            "other_donors_mean_error": float(parity_error[use != best].mean()) if len(use) > 1 else None,
        }
    group_means = [np.mean(values, axis=0) for values in scores.values()]
    means = np.mean(group_means, axis=0).tolist() if group_means else None
    return {"comparisons": receipts, "holdout_groups_scored": len(group_means), "mean_errors": means}


def affine_predictions(cal, held, prefix, guard=0):
    complete = [row for row in cal if row[prefix + "_mask"].all()]
    basis = affine_basis([pack_key(row[prefix]) for row in complete])
    checks = []
    for target in held:
        item = {"frame": target["frame"], "start_resource": target["start_resource"], "prediction": None,
                "prefix_complete": bool(target[prefix + "_mask"].all())}
        checks.append(item)
        if not item["prefix_complete"]:
            continue
        available = complete
        eligible_basis = basis
        if guard:
            available = [row for row in complete if abs(row["frame"] - target["frame"]) > guard]
            eligible_basis = affine_basis([pack_key(row[prefix]) for row in available])
        coefficient = representation(pack_key(target[prefix]), eligible_basis)
        if coefficient is None:
            continue
        sources = [row for i, row in enumerate(available) if coefficient >> i & 1]
        prediction = np.bitwise_xor.reduce([row["parity"] for row in sources])
        mask = target["parity_mask"] & np.logical_and.reduce([row["parity_mask"] for row in sources])
        item["prediction"] = {
            "sources": [[row["frame"], row["start_resource"]] for row in sources],
            "observed_bits": int(mask.sum()),
            "disagreements": int(np.count_nonzero(mask & (prediction != target["parity"]))),
        }
    return {"usable_calibration_words": len(complete), "calibration_affine_rank": len(basis),
            "checks": checks}


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--nearby-diagnostic"]):
        raise ValueError("Only --nearby-diagnostic is supported")
    followup = bool(sys.argv[1:])
    inputs = {}
    data, pages, repeated = projected_words(inputs, followup)
    rng = np.random.default_rng(SEED)
    families = {}
    packed = {}
    for label, rows in sorted(data.items()):
        cal, held = [row for row in rows if not row["held"]], [row for row in rows if row["held"]]
        permutations = np.asarray([rng.permutation(len(cal)) for _ in range(PERMUTATIONS)])
        outcomes = {}
        for prefix in ("prefix", "conditional_prefix"):
            outcomes[prefix] = {
                "nearest": nearest_predictions(cal, held, prefix, permutations) if cal else None,
                "affine": affine_predictions(cal, held, prefix),
            }
            if followup:
                outcomes[prefix]["with_nearby_guard"] = {
                    "nearest": nearest_predictions(cal, held, prefix, permutations, FRAME_GUARD) if cal else None,
                    "affine": affine_predictions(cal, held, prefix, FRAME_GUARD),
                }
        receipts = [{name: value for name, value in row.items() if not isinstance(value, np.ndarray)}
                    for row in rows]
        for field in ("prefix", "prefix_mask", "conditional_prefix", "conditional_prefix_mask", "parity", "parity_mask"):
            packed[f"{label}_{field}"] = np.packbits([row[field] for row in rows], axis=-1)
        families[label] = {"calibration_words": len(cal), "holdout_words": len(held),
                           "outcomes": outcomes, "words": receipts}
        print(label, len(cal), len(held), [(key, val["affine"]["calibration_affine_rank"])
                                          for key, val in outcomes.items()], flush=True)
    result = {
        "schema": "starlink-padding-parity-transfer/v1", "maximum_prefix_bits": PREFIX,
        "seed": SEED, "control_repetitions": PERMUTATIONS,
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "pages": pages, "formats": families,
    }
    if followup:
        result["nearby_diagnostic"] = {"frame_guard": FRAME_GUARD, "same_header_pairs": repeated}
    output = experiment_output("E074")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "padding_prefix_observations.npz", **packed)
    (output / "padding_parity_transfer.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
