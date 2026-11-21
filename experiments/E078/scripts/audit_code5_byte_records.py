"""Validate Code 5 byte records after frozen parity checks and descrambling."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_to_code21_edge import recover_systematic_erasures
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


HEADER = bytes.fromhex("00 01 f0 0f 36 0a 10 00 ff 60")
OPTIONAL_HEADER = bytes.fromhex("00 01 f0 0f 76 08 10 00 ff 60")
AFTER_RECORDS = bytes.fromhex("01 4a d6 9e 59 00 00 0a 80 5d")
RECORD_STEP = 7
RECORD_OFFSET = 66
WORD_SYSTEMATIC_BYTES = 128
START_RESOURCE = 228
BIT_LENGTH = 2048


def structure(data, override=None):
    offset = int(data[0]) if override is None else override
    base = offset + RECORD_OFFSET
    marker = data[offset:offset + RECORD_STEP] if offset >= 0 else b""
    anchor = bool(len(marker) == RECORD_STEP and marker[:2] == b"\x00\x46" and marker[4:] == b"\x00\x30\x70")
    rows = [data[x:x + RECORD_STEP] for x in range(base, base + 4 * RECORD_STEP, RECORD_STEP)
            if x >= 0 and x + RECORD_STEP <= len(data)]
    identifiers = [int.from_bytes(row[2:4], "little") for row in rows]
    tags = [row[:2].hex() for row in rows]
    ordinal = [row[4] for row in rows]
    format_tags = tags == ["403f", "40bf", "403f", "40bf"]
    paired = len(identifiers) == 4 and identifiers[0] == identifiers[1] and identifiers[2] == identifiers[3]
    low_bit = len(identifiers) == 4 and (identifiers[0] ^ identifiers[2]) == 1
    ordinal_matches = ordinal == [0, 32, 64, 96]
    prefix = data[1:11] == HEADER
    return {
        "leading_offset": int(data[0]), "tested_offset": offset, "prefix_match": prefix,
        "anchor_match": anchor, "anchor": marker.hex(),
        "record_offset": base, "all_records_available": len(rows) == 4,
        "tags": tags, "tags_match": format_tags,
        "record_ids": identifiers, "same_within_pairs": paired, "between_pairs_xor1": low_bit,
        "ordinals": ordinal, "ordinals_match": ordinal_matches,
        "last_two_bytes": [row[5:7].hex() for row in rows],
        "spans_word_boundary": base + 4 * RECORD_STEP > WORD_SYSTEMATIC_BYTES,
        "full_match": bool(prefix and anchor and format_tags and paired and low_bit and ordinal_matches),
    }


def word_parity_errors(bits, matrix):
    return int(np.count_nonzero(((bits[:1024] @ matrix) & 1) ^ bits[1024:]))


def optional_structure(data, basic):
    role = data[1:11]
    start = basic["record_offset"]
    ids = basic["record_ids"]
    alternate_order = (len(ids) == 4 and ids[0] == ids[3] and ids[1] == ids[2] and
                       basic["tags"] == ["40bf", "403f", "40bf", "403f"] and
                       basic["between_pairs_xor1"] and basic["ordinals_match"])
    variants = {HEADER: start + 28, OPTIONAL_HEADER: start}
    predicted = variants.get(role)
    return {"reverse_tag_variant": bool(basic["prefix_match"] and basic["anchor_match"] and alternate_order),
            "header_format_bytes": data[5:7].hex(), "predicted_next_record": predicted,
            "next_record_testable": predicted is not None and predicted + len(AFTER_RECORDS) <= len(data),
            "next_record_match": predicted is not None and data[predicted:predicted + len(AFTER_RECORDS)] == AFTER_RECORDS,
            "next_record_at_other_position": (
                data[start + 28:start + 28 + len(AFTER_RECORDS)] == AFTER_RECORDS if role == OPTIONAL_HEADER
                else data[start:start + len(AFTER_RECORDS)] == AFTER_RECORDS),
            "anchor_offset_controls": {str(shift): structure(data, basic["leading_offset"] + shift)["anchor_match"]
                                       for shift in (-2, -1, 1, 2)}}


def describe_word(bits, second, references, variants=False):
    all_words = bits[None] if second is None else np.vstack((bits, second))

    def extract(name, bitorder="little"):
        return np.packbits(all_words[:, :1024] ^ references[name], axis=1, bitorder=bitorder).tobytes()

    data = extract("frozen")
    result = structure(data)
    result["controls"] = {
        **{f"offset_{d:+d}": structure(data, data[0] + d)["full_match"] for d in (-2, -1, 1, 2)},
        **{f"absolute_{off}": structure(data, off)["full_match"] for off in (24, 32)},
        **{name: structure(extract(name))["full_match"] for name in ("phase_minus_one", "phase_plus_one")},
        "no_byte_bit_reversal": structure(extract("frozen", "big"))["full_match"],
    }
    result["systematic_word_sha256"] = sha256(data[:WORD_SYSTEMATIC_BYTES]).hexdigest()
    result["following_word_status"] = "verified" if second is not None else "unavailable_or_not_needed"
    if variants:
        result["additional_layouts"] = optional_structure(data, result)
    return result


def capture_rows(bits, records, matrix, refs, variants):
    keyed = {(row.get("frame", row.get("candidate_index")), row["start_resource"]): (row, b)
             for row, b in zip(records, bits, strict=True)}
    rows = []
    for (frame, resource), (old, first) in sorted(keyed.items()):
        if resource != START_RESOURCE or old["word"] != 0:
            continue
        errors = word_parity_errors(first, matrix)
        result = {"frame": frame, "source_resource": START_RESOURCE, "raw_parity_errors": errors, "result": None}
        rows.append(result)
        if errors:
            continue
        nxt = keyed.get((frame, START_RESOURCE + BIT_LENGTH))
        second = nxt[1] if nxt is not None and word_parity_errors(nxt[1], matrix) == 0 else None
        result["result"] = describe_word(first, second, refs, variants)
    return rows


def summarize(rows):
    scoring = [row["result"] for row in rows if row["result"] is not None]
    positive = [row for row in scoring if row["full_match"]]
    return {
        "frames": len(rows), "verified_first_words": len(scoring), "prefix_match": sum(row["prefix_match"] for row in scoring),
        "prefix_and_anchor_match": sum(row["prefix_match"] and row["anchor_match"] for row in scoring),
        "full_match": len(positive),
        "positive_lengths": dict(sorted(Counter(row["leading_offset"] for row in positive).items())),
        "positive_spanning": sum(row["spans_word_boundary"] for row in positive),
        "partial_prefix_and_anchor": sum(row["prefix_match"] and row["anchor_match"] and not row["all_records_available"]
                                         for row in scoring),
        "full_match_controls": dict(Counter(name for row in scoring for name, hit in row["controls"].items() if hit)),
    }


def erasure_word(values, matrix):
    bits = (values.real < 0).astype(np.uint8)
    observed = bpsk_mask(values)
    system, parity = observed[:1024], observed[1024:]
    count = int((~system).sum())
    if count > 64 or parity.sum() < 768:
        return None
    source = bits[:1024].copy()
    source[~system] = 0
    residual = ((source @ matrix) & 1) ^ bits[1024:]
    record = {"systematic_erasures": count, "parity_erasures": int((~parity).sum()),
              "folds": []}
    if count:
        first = np.isin(np.arange(1024) // 128, [0, 2, 3, 6])
        attempt = recover_systematic_erasures(matrix[~system], residual, parity, [first, ~first])
        if not attempt["validated"]:
            return None
        source[~system] = attempt["systematic_guess"]
        record["folds"] = attempt["folds"]
    elif residual[parity].any():
        return None
    predicted = (source @ matrix) & 1
    if not np.array_equal(predicted[parity], bits[1024:][parity]):
        raise ValueError("Erasure reconstruction does not match the observed parity")
    return np.r_[source, predicted], record


def bpsk_mask(values):
    return np.isclose(abs(values.real), 1, atol=2e-6, rtol=0) & (abs(values.imag) <= 2e-6)


def add_evidence(archive, frame, number, bits, source):
    key = f"frame_{frame}_word_{number}"
    archive[key + "_bits"] = np.packbits(bits)
    archive[key + "_observed"] = np.packbits(bpsk_mask(source))


def audit_archive(headers, matrix, refs, variants):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    refbits = encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48]
    signs = 1 - 2 * refbits.astype(float)
    rows = []
    evidence = {}
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        source = archive["yDataDec"]
        if source.shape != (1009, 301, 1024):
            raise ValueError("Unexpected Qin archive size")
        for block_start in range(0, len(headers), source.chunks[0]):
            stop = min(block_start + source.chunks[0], len(headers))
            data = as_complex(source[block_start:stop, 1:6, :])[..., NONPILOT % 1024]
            for frame, grid in enumerate(data, block_start):
                header = headers[frame]
                allocation = header["h1"]["layout20"]["entries"][0]
                code5 = allocation["mcs_candidate"] == 5
                rows.append({"frame": frame, "source_resource": START_RESOURCE, "raw_parity_errors": None,
                             "result": None, "code5_first_allocation": code5})
                if not code5 or allocation["quantity"] < 1:
                    continue
                received = remove_header_gain(grid * rotation[:len(grid)], signs)

                def samples(n):
                    start = START_RESOURCE + n * BIT_LENGTH
                    return received[start:start + BIT_LENGTH]

                def hard_word(n):
                    values = samples(n)
                    if not bpsk_mask(values).all():
                        return None
                    return (values.real < 0).astype(np.uint8)

                bit = hard_word(0)
                if bit is None:
                    if variants:
                        restored = erasure_word(samples(0), matrix)
                        if restored is not None:
                            first, details = restored
                            lead = np.packbits(first[:8] ^ refs["frozen"][:8], bitorder="little")[0].item()
                            more = erasure_word(samples(1), matrix) if (
                                allocation["quantity"] >= 2 and lead + 104 > WORD_SYSTEMATIC_BYTES) else None
                            rows[-1]["erasure_result"] = {
                                "first_word": details, "second_word": more[1] if more else None,
                                "result": describe_word(first, more[0] if more else None, refs, variants),
                            }
                            add_evidence(evidence, frame, 0, first, samples(0))
                            if more is not None:
                                add_evidence(evidence, frame, 1, more[0], samples(1))
                    continue
                errors = word_parity_errors(bit, matrix)
                rows[-1]["raw_parity_errors"] = errors
                if errors:
                    continue
                lead = np.packbits(bit[:8] ^ refs["frozen"][:8], bitorder="little")[0].item()
                required = 104 if variants else 94
                nxt = hard_word(1) if allocation["quantity"] >= 2 and lead + required > WORD_SYSTEMATIC_BYTES else None
                if nxt is not None and word_parity_errors(nxt, matrix):
                    nxt = None
                rows[-1]["result"] = describe_word(bit, nxt, refs, variants)
                if variants:
                    add_evidence(evidence, frame, 0, bit, samples(0))
                    if nxt is not None:
                        add_evidence(evidence, frame, 1, nxt, samples(1))
            if block_start % 130 == 0:
                print("Audited archived Code 5 initial frames up to", stop, flush=True)
    pages = [reader.receipts[key] for key in sorted(reader.receipts)]
    return rows, pages, evidence


def main():
    require_run_context()
    if sys.argv[1:] not in ([], ["--variants-and-erasures"]):
        raise ValueError("Only --variants-and-erasures is supported")
    variants = bool(sys.argv[1:])
    inputs = {}
    pn_model = read(checked_artifact("v071-pn-parity-order", "E073", "pn_parity_order.json", inputs))["pn_model"]
    old_run, ut_run = "v010-code5-parity-02", "v022-ut-replay"
    previous = read(checked_artifact(old_run, "E012", "code5_parity.json", inputs))
    present = read(checked_artifact(ut_run, "E014", "ut_transfer.json", inputs))
    with np.load(checked_artifact(old_run, "E012", "code5_observations.npz", inputs), allow_pickle=False) as data:
        matrix, old_bits = data["parity_generator"], data["hard_bits"]
    with np.load(checked_artifact(ut_run, "E014", "ut_observations.npz", inputs), allow_pickle=False) as data:
        ut_bits = data["hard_code5_words"]
    pn = pn_bits()
    coords = np.arange(1024)
    refs = {
        "frozen": equality_predictions(coords, pn, pn_model),
        "phase_minus_one": equality_predictions(coords, pn, {**pn_model, "phase": pn_model["phase"] - 1}),
        "phase_plus_one": equality_predictions(coords, pn, {**pn_model, "phase": pn_model["phase"] + 1}),
    }
    raw = {
        "legacy_480": capture_rows(old_bits, previous["words"], matrix, refs, variants),
        "ut_250": capture_rows(ut_bits, present["words"], matrix, refs, variants),
    }
    headers = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))["frames"]
    raw["qin_archive"], pages, evidence = audit_archive(headers, matrix, refs, variants)
    for key, data in raw.items():
        print(key, summarize(data), flush=True)
    out = experiment_output("E078")
    out.mkdir(parents=True, exist_ok=True)
    if evidence:
        np.savez_compressed(out / "archived_code5_words.npz", **evidence)
    report = {"schema": "starlink-code5-byte-records/v1",
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE, "pages": pages}, "raw": raw,
              "summary": {name: summarize(rows) for name, rows in raw.items()}}
    (out / "code5_byte_records.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
