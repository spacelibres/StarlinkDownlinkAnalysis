"""Recover a Code 5 directory CRC and verify its first record boundary."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_archive_code5_recovery import normalize_axis
from audit_code5_byte_records import BIT_LENGTH, START_RESOURCE, bpsk_mask, erasure_word
from audit_code5_frame_count import descramble
from audit_code5_to_code21_edge import carrier_axis, recover_systematic_erasures
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


WIDTH = 24
MASK = (1 << WIDTH) - 1
FIT_LENGTHS = (24, 32)
SEED = 2026112192
REFLECT_BYTE = bytes(int(f"{n:08b}"[::-1], 2) for n in range(256))
FIRST_SIGNATURE = bytes.fromhex("00 01 f0 0f")
SECOND_MARKER = bytes.fromhex("08 5d")


def remainder(value, divisor):
    if divisor <= 0:
        raise ValueError("Polynomial divisor must be nonzero")
    while value.bit_length() >= divisor.bit_length():
        value ^= divisor << (value.bit_length() - divisor.bit_length())
    return value


def gcd(left, right):
    while right:
        left, right = right, remainder(left, right)
    return left


def view(data, convention):
    if convention == "msb_first":
        return int.from_bytes(data, "big")
    if convention == "reversed_trailer_bytes":
        data = data[:-3] + data[-3:][::-1]
    return int.from_bytes(data.translate(REFLECT_BYTE), "big")


def difference_gcds(words, convention):
    by_length = defaultdict(set)
    for header in words:
        by_length[len(header)].add(header)
    result = {}
    for length, unique in sorted(by_length.items()):
        vectors = [view(header, convention) for header in sorted(unique)]
        divisor = 0
        for vector in vectors[1:]:
            divisor = gcd(divisor, vectors[0] ^ vector)
        result[length] = {"distinct": len(vectors), "gcd": divisor}
    return result


def binary_solution(columns, target, dimension):
    equations = {}
    for j in range(dimension):
        row = sum(((column >> j) & 1) << k for k, column in enumerate(columns))
        rhs = (target >> j) & 1
        while row:
            pivot = row.bit_length() - 1
            if pivot not in equations:
                equations[pivot] = row, rhs
                break
            known, known_rhs = equations[pivot]
            row ^= known
            rhs ^= known_rhs
        else:
            if rhs:
                return None
    value = 0
    for pivot, (row, rhs) in sorted(equations.items()):
        value |= (((row & value).bit_count() & 1) ^ rhs) << pivot
    return value, len(equations)


def fit_affine(words, poly, start=0):
    residues = defaultdict(set)
    for word in words:
        residues[len(word) - start].add(remainder(view(word[start:], "lsb_first"), poly))
    if len(residues) < 2 or any(len(x) != 1 for x in residues.values()):
        return None
    base, *others = sorted((n, next(iter(r))) for n, r in residues.items())
    shift = lambda x, n: remainder(x << (8 * n), poly)
    columns = [sum((shift(1 << i, n) ^ shift(1 << i, base[0])) << (WIDTH * j)
                   for j, (n, _) in enumerate(others)) for i in range(WIDTH)]
    target = sum((r ^ base[1]) << (WIDTH * j) for j, (_, r) in enumerate(others))
    solution = binary_solution(columns, target, WIDTH * len(others))
    if solution is None:
        return None
    seed, rank = solution
    return {"seed": seed, "rank": rank, "residue": (base[1] ^ shift(seed, base[0]))}


def crc_bytes(prefix, poly, seed, residue):
    length = len(prefix) + 3
    coded_zero = view(prefix, "lsb_first") << WIDTH
    value = remainder(coded_zero ^ (seed << (8 * length)), poly) ^ residue
    return value.to_bytes(3, "big").translate(REFLECT_BYTE)


def raw_record_set(path, rows, key, pn, generator):
    frames = defaultdict(dict)
    with np.load(path, allow_pickle=False) as z:
        for m, bits in zip(rows, z[key], strict=True):
            number = m["word"]
            if number not in (0, 1) or m["start_resource"] != START_RESOURCE + number * BIT_LENGTH:
                continue
            if not np.array_equal((bits[:1024] @ generator) & 1, bits[1024:]):
                raise ValueError("Raw Code 5 witness has nonzero outer parity")
            frame = m.get("frame", m.get("candidate_index"))
            frames[frame][number] = (descramble(bits, pn), np.ones(1024, bool))
    return frames


def archive_first_and_prior(path, extension, pn, generator):
    frames, stored = {}, {}
    for file in (path, extension):
        with np.load(file, allow_pickle=False) as z:
            for key in z.files:
                if not key.endswith("_bits"):
                    continue
                frame, word = (int(key.split("_")[i]) for i in (1, 3))
                bits = np.unpackbits(z[key])
                if not np.array_equal((bits[:1024] @ generator) & 1, bits[1024:]):
                    raise ValueError(f"Archive Code 5 outer parity changed: {key}")
                seen = np.unpackbits(z[key.removesuffix("_bits") + "_observed"]).astype(bool)
                if word == 0:
                    if frame in frames:
                        raise ValueError("Duplicate predecessor first word")
                    frames[frame] = (descramble(bits, pn), seen[:1024])
                elif word == 1:
                    stored[frame] = (descramble(bits, pn), seen[:1024])
    return frames, stored


def recover_partial_parity(values, generator):
    """Retain a truncated word only when both parity folds predict its erasure."""
    seen = bpsk_mask(values)
    unknown = ~seen[:1024]
    observed = seen[1024:]
    if unknown.sum() != 1 or observed.sum() >= 768:
        return None
    bits = (values.real < 0).astype(np.uint8)
    systematic = bits[:1024]
    systematic[unknown] = 0
    syndrome = ((systematic @ generator) & 1) ^ bits[1024:]
    first = np.isin(np.arange(1024) // 128, [0, 2, 3, 6])
    receipt = recover_systematic_erasures(generator[unknown], syndrome, observed, [first, ~first])
    if not receipt["validated"]:
        return None
    systematic[unknown] = receipt["systematic_guess"]
    parity = (systematic @ generator) & 1
    if np.any((bits[1024:] ^ parity) & observed):
        raise ValueError("Partial Code 5 evidence failed the final observed parity check")
    bits[1024:] = parity
    return bits, {"erased_systematic_position": int(np.flatnonzero(unknown)[0]),
                  "observed_parity": int(observed.sum()), **receipt}


def extend_second_words(count, generator, pn, stored):
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 7)) for k in NONPILOT]).conj()
    start = START_RESOURCE + BIT_LENGTH
    clipped = carrier_axis(start, BIT_LENGTH) == 511
    successes, failed, truncation = {}, [], {}
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        data = archive["yDataDec"]
        if data.shape != (1009, 301, 1024):
            raise ValueError("Published archive dimensions changed")
        for block in range(0, count, data.chunks[0]):
            last = min(block + data.chunks[0], count)
            grids = as_complex(data[block:last, 1:6, :])[..., NONPILOT % 1024]
            for i, grid in enumerate(grids, block):
                if i in stored:
                    continue
                symbols = remove_header_gain(grid * rotation, signs)[start:start + BIT_LENGTH]
                corrected = normalize_axis(symbols)
                if corrected is None:
                    failed.append(i)
                    continue
                corrected[clipped] = 0
                recovered = erasure_word(corrected, generator)
                if recovered is None:
                    recovered = recover_partial_parity(corrected, generator)
                    if recovered is not None:
                        truncation[i] = recovered[1]
                if recovered is None:
                    failed.append(i)
                    continue
                bits, _ = recovered
                seen = bpsk_mask(corrected)
                stored[i] = (descramble(bits, pn), seen[:1024])
                successes[i] = (np.packbits(bits), np.packbits(seen))
    pages = [reader.receipts[p] for p in sorted(reader.receipts)]
    return successes, failed, pages, truncation


def report_row(i, first, second, poly, affine, rng):
    data, observed = first
    length = data[0]
    if not 20 <= length <= 60 or data[1:5] != FIRST_SIGNATURE:
        raise ValueError("Unexpected Code 5 directory prefix")
    header = data[:length]
    actual = header[-3:]
    expected = crc_bytes(header[:-3], poly, affine["seed"], affine["residue"])
    altered = bytearray(header)
    error = int(rng.integers(8, (length - 3) * 8))
    altered[error // 8] ^= 1 << (error % 8)
    corrupted = crc_bytes(altered[:-3], poly, affine["seed"], affine["residue"])
    span = int.from_bytes(data[5:7], "little") >> 4
    row = {"frame": i, "directory_length": length, "decoded_record_span": span,
           "observed_directory_bits": int(observed[:length * 8].sum()),
           "observed_crc_bits": int(observed[(length - 3) * 8:length * 8].sum()),
           "expected_crc": expected.hex(), "crc_pass": actual == expected,
           "crc_observation": actual.hex(), "flipped_directory_bit": error, "flipped_pass": actual == corrupted,
           "next_marker_readable": second is not None}
    if second is not None:
        joined = data + second[0]
        jointly_observed = np.concatenate((observed, second[1]))
        end = length + span
        row.update(predicted_marker_offset=end, marker=joined[end:end + 2].hex(),
                   marker_bits_observed=int(jointly_observed[end * 8:(end + 2) * 8].sum()),
                   marker_pass=joined[end:end + 2] == SECOND_MARKER,
                   shifted_marker_pass={shift: joined[end + shift:end + shift + 2] == SECOND_MARKER
                                        for shift in (-2, -1, 1, 2)},
                   always_163_pass=joined[length + 163:length + 165] == SECOND_MARKER)
    return row


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model = read(artifact("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    legacy = "v010-code5-parity-02"
    old_iq = artifact(legacy, "E012", "code5_observations.npz")
    with np.load(old_iq, allow_pickle=False) as z:
        generator = z["parity_generator"]
    reference = "v078-archive-code5-links"
    archive_rows = read(artifact(reference, "E080", "archive_code5_recovery.json"))["frames"]
    old_archive = artifact("v076-code5-byte-words", "E078", "archived_code5_words.npz")
    frames, second = archive_first_and_prior(old_archive, artifact(reference, "E080", "recovered_code5_words.npz"),
                                             pn, generator)
    training = [r["archive_index"] for r in archive_rows if r["calibration"] and
                frames[r["archive_index"]][0][0] in FIT_LENGTHS]
    headers = [frames[i][0][:frames[i][0][0]] for i in training]
    orientations = {code: difference_gcds(headers, code)
                    for code in ("lsb_first", "msb_first", "reversed_trailer_bytes")}
    polys = {r["gcd"] for r in orientations["lsb_first"].values()}
    if len(polys) != 1:
        raise ValueError("Calibration does not define a common polynomial")
    poly = polys.pop()
    if poly.bit_length() != WIDTH + 1:
        raise ValueError("Calibration does not define a degree-24 CRC")
    affine = fit_affine(headers, poly)
    if affine is None:
        raise ValueError("No degree-24 CRC with compatible encoding")
    split_ids = {r["archive_index"]: (r["group"], r["calibration"]) for r in archive_rows}
    qin_headers = read(artifact("v034-qin-header-survey", "E036", "qin_header_survey.json"))["frames"]
    allocations = [r["h1"]["layout20"]["entries"][0] for r in qin_headers]
    if len(frames) != len(split_ids) or len(qin_headers) != len(frames) or any(
        not r["h1"]["crc_ok"] or a["mcs_candidate"] != 5 or a["quantity"] < 2
        for r, a in zip(qin_headers, allocations, strict=True)
    ):
        raise ValueError("Every archive frame should expose a second Code 5 resource area")
    old_second_ids = sorted(second)
    recovered, failed, pages, truncation = extend_second_words(len(qin_headers), generator, pn, second)
    rng = np.random.default_rng(SEED)
    raw = "v022-ut-replay"
    old_words = raw_record_set(old_iq, read(artifact(legacy, "E012", "code5_parity.json"))["words"],
                               "hard_bits", pn, generator)
    ut_words = raw_record_set(artifact(raw, "E014", "ut_observations.npz"),
                              read(artifact(raw, "E014", "ut_transfer.json"))["words"],
                              "hard_code5_words", pn, generator)
    rows = []
    for i, first in sorted(frames.items()):
        row = report_row(i, first, second.get(i), poly, affine, rng)
        row["group"], row["calibration"] = split_ids[i]
        rows.append(row)
    record = {"schema": "starlink-code5-directory-checks/v1",
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()},
              "crc_fit_lengths": FIT_LENGTHS, "crc_fit_frames": training, "orientation_gcds": orientations,
              "crc_polynomial_including_top_bit": poly, "affine_convention": affine,
              "fit_skipping_length_byte": fit_affine(headers, poly, start=1),
              "bit_flip_control_seed": SEED, "archive": rows, "raw": {
                  label: [report_row(i, frame[0], frame.get(1), poly, affine, rng) for i, frame in sorted(data.items())]
                  for label, data in (("480MSps", old_words), ("250MSps", ut_words))},
              "second_word_from_e078": old_second_ids, "new_second_word_count": len(recovered),
              "second_word_unresolved_frames": failed,
              "second_word_truncated_parity_recovery": truncation,
              "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE, "pages": pages}}
    out = experiment_output("E091")
    out.mkdir(parents=True, exist_ok=True)
    selected = sorted(recovered)
    np.savez_compressed(out / "code5_directory_following_words.npz", frame_index=np.array(selected),
                        bits=np.array([recovered[i][0] for i in selected]),
                        observed=np.array([recovered[i][1] for i in selected]))
    (out / "code5_directory_checks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("CRC polynomial", hex(poly), "seed", hex(affine["seed"]), "residue", hex(affine["residue"]),
          "archive CRC", Counter(x["crc_pass"] for x in rows),
          "archive next marker", Counter(x.get("marker_pass") for x in rows), flush=True)


if __name__ == "__main__":
    main()
