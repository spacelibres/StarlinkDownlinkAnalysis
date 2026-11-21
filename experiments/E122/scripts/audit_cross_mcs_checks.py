"""Transfer frozen MCS 66 and 70 checks across modulation alphabets."""

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_directory_records import pilot_archive_windows
from audit_code5_gain_transfer import decisions, four_corner_slots
from audit_leo_code5_projection import nominal_capture_windows, selected_windows
from audit_pn_padding_starts import CacheOnly
from audit_qin_exemplars import remove_header_gain
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_reference
from recover_mcs66_systematic_checks import LIFT, project_checks, totals
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


BIT_LENGTH = 16384
FORMATS = (66, 70, 71, 75, 78, 81, 85, 88, 92, 93)
EXPECTED = {66: 66, 70: 70, 85: 66, 88: 70}
BASE = {66: 12288, 70: 12800}


def paired_syndromes(bits, mask, groups, support, alternative):
    direct, direct_seen = project_checks(bits, mask, support)
    control, control_seen = project_checks(bits, mask, alternative)
    shared = direct_seen & control_seen
    return totals(direct, shared, groups), totals(control, shared, groups)


def visible_terms(start, support, bits_per_resource, carrier_mask):
    indices = (start + support // bits_per_resource) % len(carrier_mask)
    return carrier_mask[indices].sum(axis=-1)


def checks(artifact):
    paths = {
        66: artifact("v081-mcs66-systematic-checks", "E083", "mcs66_partial_checks.npz"),
        70: artifact("v118-mcs70-sparse", "E120", "mcs70_sparse_checks.npz"),
    }
    families = {}
    for code, path in paths.items():
        with np.load(path, allow_pickle=False) as npz:
            matrix = np.unpackbits(npz["check_matrix_packed"], axis=1)
        if matrix.shape != (2 * LIFT, BIT_LENGTH) or np.any(matrix.sum(axis=1) != 14):
            raise ValueError(f"Unexpected check matrix for MCS {code}")
        families[code] = np.nonzero(matrix)[1].reshape(2, LIFT, 14)
    return families


def archive_observations(source, gain, layout, models):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    prefix = np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)]
    signs = 1 - 2 * encode_block(prefix)[:48].astype(float)
    cohorts = defaultdict(list)

    @lru_cache(maxsize=1)
    def block(first):
        end = min(first + source.chunks[0], len(source))
        return as_complex(source[first:end, 1:, :])[..., NONPILOT % 1024]

    for row, geometry in zip(gain["frames"], layout["word_windows_and_onsets"], strict=True):
        frame = row["frame"]
        if frame != geometry["frame"]:
            raise ValueError("Archive frame order differs")
        target = []
        for obs, alloc in zip(row["allocations"], geometry["allocations"], strict=True):
            code = alloc["label"]
            if obs["start"] != alloc["start"] or obs["label"] != code:
                raise ValueError("Archive allocation coordinates differ")
            if code in FORMATS:
                target.append((alloc, alloc["count"] - 1, "terminal"))
                if alloc["count"] > 1:
                    target.append((alloc, 0, "nonterminal"))
        if not target:
            continue
        first = frame // source.chunks[0] * source.chunks[0]
        flat = remove_header_gain(block(first)[frame - first] * rotation, signs)
        axis = complex(*row["axis"])
        corners = four_corner_slots(flat, axis)
        for alloc, word, role in target:
            code = alloc["label"]
            resource = alloc["start"] + word * alloc["width_resources"]
            stop = layout["marker_systematic_ends"][str(code)]
            bits, seen, _, _ = decisions(flat, axis, {**alloc, "start": resource, "count": 1},
                                         models, stop, corners)
            length = BIT_LENGTH + (alloc["bits_per_symbol"] == 5)
            if bits.shape != (1, length):
                raise ValueError(f"Unexpected serialized width in MCS {code}")
            coords = {"frame": frame, "group": row["group"], "calibration": row["calibration"],
                      "resource": resource, "word": word, "label": code}
            cohorts[code, role].append((coords, np.packbits(bits[0] & seen[0]), np.packbits(seen[0])))
    return cohorts


def raw_observations(artifact, labels):
    source = artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    metadata = read(artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_extraction.json"))
    points = np.array([complex(p["i"], p["q"]) for p in labels["cross32_mapping"]]) / np.sqrt(20)
    mapping = np.array([p["label"] for p in labels["cross32_mapping"]])
    result = defaultdict(list)
    with np.load(source, allow_pickle=False) as observed:
        if not np.allclose(points, observed["cross32_points"]):
            raise ValueError("Frozen 32QAM point order differs")
        for code in (71, 78, 85, 88, 92):
            values = observed[f"mcs{code}_symbols"]
            frames = observed[f"mcs{code}_frames"]
            bits = labels_as_bits(mapping[nearest_symbols(values, points)], 5).reshape(len(values), -1)
            if bits.shape[1] != BIT_LENGTH + 1 or not np.isfinite(values).all():
                raise ValueError("Invalid raw 32QAM observations")
            packed = np.packbits(bits, axis=1)
            mask = np.packbits(np.ones(BIT_LENGTH + 1, dtype=bool))
            cursor = 0
            for region in metadata["regions"]:
                if region["format"] != code:
                    continue
                frame = region["frame"]
                count = region["count"]
                if not np.array_equal(frames[cursor:cursor + count], np.full(count, frame)):
                    raise ValueError("Raw 32QAM allocation order differs")
                for word in range(count):
                    role = "terminal" if word == count - 1 else "nonterminal"
                    coords = {"frame": frame, "group": frame, "calibration": frame in metadata["calibration_frames"],
                              "resource": region["start"] + word * values.shape[1], "word": word, "label": code}
                    result[code, role].append((coords, packed[cursor + word], mask))
                cursor += count
            if cursor != len(values):
                raise ValueError("Missing raw 32QAM allocation")
    return result, metadata["split_seed"]


def score_cohort(bits, mask, rows, matrices):
    groups = np.array([row["group"] for row in rows])
    result = {}
    for code, families in matrices.items():
        by_family = []
        for support in families:
            value, seen = project_checks(bits, mask, support)
            score = {"direct": totals(value, seen, groups)}
            systematic = support < BASE[code]
            shift = support.copy()
            shift[systematic] = support[systematic] // LIFT * LIFT + (support[systematic] + 1) % LIFT
            comp = paired_syndromes(bits, mask, groups, support, shift)
            score["systematic_shift_control"] = dict(zip(("direct", "shifted"), comp, strict=True))
            if rows[0]["label"] not in (66, 70):
                comp = paired_syndromes(bits, mask, groups, support, support + 1)
                score["serial_shift_control"] = dict(zip(("direct", "shifted"), comp, strict=True))
            by_family.append(score)
        result[str(code)] = by_family
    return result


def evaluate(cohorts, matrices, packed, origin):
    summary = {}
    for (code, role), observations in sorted(cohorts.items()):
        rows, bits, mask = zip(*observations, strict=True)
        key = f"{code}_{role}"
        packed[f"{origin}_{key}_bits"] = bits = np.stack(bits)
        packed[f"{origin}_{key}_seen"] = mask = np.stack(mask)
        bits = np.unpackbits(bits, axis=1)
        mask = np.unpackbits(mask, axis=1).astype(bool)
        reserved = np.array([not row["calibration"] for row in rows])
        result = {"rows": rows, "expected_matrix": EXPECTED.get(code)}
        for name, use in (("reserved", reserved), ("calibration", ~reserved)):
            selected = [r for r, keep in zip(rows, use, strict=True) if keep]
            if selected:
                result[name] = {"words": len(selected), "groups": len({r["group"] for r in selected}),
                                "matrices": score_cohort(bits[use], mask[use], selected, matrices)}
        summary[key] = result
    return summary


def leo_projection(catalogue, pilot_receipt, matrices, external):
    windows = (selected_windows(external) + nominal_capture_windows(catalogue)
               + list(pilot_archive_windows(pilot_receipt, external)))
    cached = {}
    for window in windows:
        mask = np.isin(NONPILOT, window["carriers"])
        key = mask.tobytes()
        if key not in cached:
            coverage = {}
            for code, families in matrices.items():
                for width in (4, 5):
                    counts = np.zeros(15, dtype=np.int64)
                    support = families.reshape(-1, 14)
                    for start in range(len(mask)):
                        counts += np.bincount(visible_terms(start, support, width, mask), minlength=len(counts))
                    coverage[f"{code}_{width}"] = counts.tolist()
            cached[key] = coverage
        yield {**window, "row_visible_terms_across_all_codeword_origins": cached[key]}


def main():
    require_run_context()
    hashes, external, packed = {}, {}, {}

    def artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, hashes)

    matrices = checks(artifact)
    gain = read(artifact("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    layout = read(artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json"))
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    models = constellation_models(labels, hashes)
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        cohorts = archive_observations(archive["yDataDec"], gain, layout, models)
    print("Qin first/nonterminal and terminal word extraction", sum(map(len, cohorts.values())), flush=True)
    result = {"schema": "starlink-cross-mcs-checks/v1", "candidate_codeword_bits": BIT_LENGTH,
              "archive": evaluate(cohorts, matrices, packed, "archive")}
    for label, matrix in ((85, 66), (88, 70)):
        row = result["archive"][f"{label}_nonterminal"]["reserved"]
        print(f"MCS {label}, {row['words']} held nonterminal words, {row['groups']} groups:",
              [r["direct"]["group_accuracy"] for r in row["matrices"][str(matrix)]], flush=True)
    raw, seed = raw_observations(artifact, labels)
    result["raw250"] = evaluate(raw, matrices, packed, "raw250")
    catalogue = read(artifact("v069-leo-recording-coverage", "E071", "leo_recording_coverage.json"))
    pilot = read(artifact("v097-leo-pilot-archive", "E099", "leo_pilot_archive.json"))
    result["leo"] = list(leo_projection(catalogue, pilot, matrices, external))
    result.update(raw250_split_seed=seed, archive_split_seed=2026112177, leo_input_sha256=external,
                  artifact_input_sha256={path.relative_to(ROOT).as_posix(): digest for path, digest in hashes.items()},
                  archive_source={"url": URL, "etag": ETAG, "bytes": SIZE,
                                  "pages": [reader.receipts[k] for k in sorted(reader.receipts)]})
    out = experiment_output("E122")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "cross_mcs_observations.npz", **packed)
    (out / "cross_mcs_checks.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
