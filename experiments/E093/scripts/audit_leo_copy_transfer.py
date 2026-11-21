"""Test the original LEO upper-edge copy hypothesis in CRC-checked Code 5."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code5_directory_checks import archive_first_and_prior, crc_bytes, raw_record_set
from audit_code5_frame_count import RELATIVE_BIT
from audit_leo_early_symbol_roles import CODE5_START, REFERENCE, bin_coordinates
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits, pn_reference


SYMBOL = 2
BINS = [480, 486]
RESOURCES = tuple(map(int, bin_coordinates(BINS, SYMBOL)))
POSITIONS = tuple(r - CODE5_START for r in RESOURCES)
LEO_STUDY = Path("reports/2026_09_29_all_track_symbols")
DS8_HASHES = Path("reports/figures/2026-09-28-ds7-ds8/input-hashes.sha256")


def record_roles(length, starts, width):
    roles = []
    for pos in POSITIONS:
        rel = pos - 8 * length
        role = "unclassified"
        if RELATIVE_BIT <= rel < RELATIVE_BIT + 32:
            role = "frame_count"
        for start in starts:
            if start <= rel < start + width:
                role = f"trajectory_{start}"
        roles.append((rel, role))
    return roles


def tally(rows):
    return {"pairs": len(rows), "equal": sum(row["equal"] for row in rows),
            "observed_pairs": sum(row["observed_both"] for row in rows),
            "observed_equal": sum(row["equal"] and row["observed_both"] for row in rows)}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def external_inventory():
    root = EXTERNAL_LEO_ROOT
    paths = [LEO_STUDY / x for x in ("REFERENCE_COPY_TRANSFER.md", "reference_copy_transfer.py", "HEADER_STATE_TRANSFER.md")]
    paths += [DS8_HASHES, Path("reports/2026-09-28-ds7-ds8-signal-structure.md"),
              Path("reports/2026_10_18_dataset_manifest_archive/catalogue.json")]
    hashes = {path.as_posix(): file_hash(root / path) for path in paths}
    report_lines = (root / paths[0]).read_text(encoding="utf-8").splitlines()
    declaration = next(line for line in report_lines if "equal sign" in line)
    if "OFDM symbol 2, FFT carriers 480 and 486, equal sign" not in declaration:
        raise ValueError("The frozen LEO relation changed")
    ds8 = {}
    for line in (root / DS8_HASHES).read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = root / name
        observed = file_hash(path) if path.is_file() else None
        ds8[name] = {"expected": expected, "observed": observed}
    local = root / "reports/2026_09_28_sequence_semantics/local"
    soft_caches = {name: file_hash(local / name) if (local / name).is_file() else None
                   for name in ("header-reference-0-77.npz", "S13-soft.npz", "S22-soft.npz", "S23-soft.npz",
                                "DS9-middle-soft.npz", "DS9-last-soft.npz")}
    return {"sources_sha256": hashes, "ds7_ds8_hash_register": ds8, "soft_caches_sha256": soft_caches}


def codeword_rows(words, receipts, poly, affine, reference_xor):
    if set(words) != {r["frame"] for r in receipts}:
        raise ValueError("Directory receipt membership differs from recovered words")
    rows = []
    for r in receipts:
        frame = r["frame"]
        data, seen = words[frame]
        length = data[0]
        observed_directory = int(seen[:8 * length].sum())
        if (not r["crc_pass"] or length != r["directory_length"]
                or observed_directory != r["observed_directory_bits"]
                or not seen[8 * (length - 3):8 * length].all()):
            raise ValueError(f"Directory observation or CRC receipt changed in frame {frame}")
        expected = crc_bytes(data[:length - 3], poly, affine["seed"], affine["residue"])
        if expected != data[length - 3:length] or expected.hex() != r["expected_crc"]:
            raise ValueError(f"Directory CRC changed in frame {frame}")
        payload_bits = [(data[i // 8] >> (i % 8)) & 1 for i in POSITIONS]
        signs_equal = (payload_bits[0] ^ payload_bits[1]) == reference_xor
        rows.append({"frame": frame, "length": length, "equal": signs_equal, "data_bits": payload_bits,
                     "directory_has_erasure": observed_directory != 8 * length,
                     "observed_both": bool(seen[list(POSITIONS)].all()), "archive_group": r.get("group")})
    return rows


def summarize_by_length(rows, starts, width):
    groups = defaultdict(list)
    for row in rows:
        groups[row["length"]].append(row)
    return {length: {**tally(group), "record_coordinates": record_roles(length, starts, width)}
            for length, group in sorted(groups.items())}


def main():
    require_run_context()
    inputs = {}

    def artifact(run, experiment, name):
        return checked_artifact(run, experiment, name, inputs)

    external = external_inventory()
    old = "v010-code5-parity-02"
    raw_iq = artifact(old, "E012", "code5_observations.npz")
    with np.load(raw_iq, allow_pickle=False) as data:
        generator = data["parity_generator"]
    model = read(artifact("v071-pn-parity-order", "E073", "pn_parity_order.json"))["pn_model"]
    pn = equality_predictions(np.arange(1024), pn_bits(), model)
    archive, _ = archive_first_and_prior(
        artifact("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
        artifact("v078-archive-code5-links", "E080", "recovered_code5_words.npz"), pn, generator,
    )
    directory = read(artifact("v089-code5-directory-final", "E091", "code5_directory_checks.json"))
    e092 = read(artifact("v090-code5-trajectories", "E092", "code5_record_trajectories.json"))
    hdr = read(artifact("v034-qin-header-survey", "E036", "qin_header_survey.json"))["frames"]
    for r in hdr:
        entries = r["h1"]["layout20"]["entries"]
        if (not r["h1"]["crc_ok"] or not r["h2"]["crc_ok"] or entries[0]["mcs_candidate"] != 5
                or r["h1"]["needed_resources"] != CODE5_START or r["h2_start"] <= max(RESOURCES)):
            raise ValueError(f"Unexpected archive allocation in frame {r['frame']}")
    if set(archive) != {r["frame"] for r in hdr}:
        raise ValueError("Archive Code 5 and header inventories differ")

    inputs[REFERENCE] = file_hash(REFERENCE)
    q = np.exp(0.5j * np.pi * loadmat(REFERENCE)["referenceTemplateRotations"][BINS, SYMBOL - 1])
    ratios = q.conj() * np.array([pn_reference(k, np.array([SYMBOL]))[0] for k in BINS])
    if np.any(abs(ratios.imag) > 1e-10):
        raise ValueError("Unexpected quadrature after converting from analytical to Qin reference")
    reference_xor = int((ratios.real[0] < 0) ^ (ratios.real[1] < 0)) ^ int(pn[POSITIONS[0]] ^ pn[POSITIONS[1]])
    poly = directory["crc_polynomial_including_top_bit"]
    affine = directory["affine_convention"]
    rows = codeword_rows(archive, directory["archive"], poly, affine, reference_xor)
    held = set(e092["held_frames"])
    first_78 = [r for r in rows if r["frame"] < 78]
    later = [r for r in rows if r["frame"] >= 78]
    held_later = [r for r in later if r["frame"] in held]
    starts = [family[0] for family in e092["families"]]
    width = e092["searched_width"]

    snapshot = read(artifact("v032-qin-snapshot", "E034", "qin_snapshot_source.json"))
    by_frame = {r["frame"]: r for r in rows}
    hard = []
    for group in snapshot["hard_groups"]:
        with np.load(artifact("v032-qin-snapshot", "E034", f"qin_hard_{group}.npz"), allow_pickle=False) as data:
            for frame, symbols in zip(data["frame"], data["symbols"], strict=True):
                values = symbols[SYMBOL - 1, BINS] * q.conj()
                valid = np.all((abs(values.imag) < 0.05) & (abs(abs(values.real) - 1) < 0.05))
                equal = bool((values.real[0] >= 0) == (values.real[1] >= 0)) if valid else None
                hard.append({"frame": int(frame), "observed_equal": equal, "decoded_equal": by_frame[int(frame)]["equal"]})

    raw = {}
    for key, run, exp, words, meta, array in (
        ("480MSps", old, "E012", "code5_observations.npz", "code5_parity.json", "hard_bits"),
        ("250MSps", "v022-ut-replay", "E014", "ut_observations.npz", "ut_transfer.json", "hard_code5_words"),
    ):
        by_word = raw_record_set(artifact(run, exp, words), read(artifact(run, exp, meta))["words"], array, pn, generator)
        first_words = {frame: codewords[0] for frame, codewords in by_word.items() if 0 in codewords}
        captured = codeword_rows(first_words, directory["raw"][key], poly, affine, reference_xor)
        raw[key] = {"summary": tally(captured), "rows": captured}

    result = {"schema": "starlink-leo-copy-transfer/v1", "external": external,
              "inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
              "coordinates": {"symbol": SYMBOL, "bins": BINS, "resources": RESOURCES, "systematic": POSITIONS,
                              "analytical_to_qin_real_signs": ratios.real.round().astype(int).tolist(),
                              "descrambled_xor_predicting_equal_signs": reference_xor},
              "archive_partitions": {"full": tally(rows), "leo_first_78": tally(first_78),
                                     "outside_leo_78": tally(later), "e092_held_outside_leo_78": tally(held_later)},
              "archive_by_length": summarize_by_length(rows, starts, width), "archive_rows": rows,
              "archive_original_hard": hard, "wideband_replay": raw}
    target = experiment_output("E093")
    target.mkdir(parents=True, exist_ok=True)
    (target / "leo_copy_transfer.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["archive_partitions"])
    print("Raw first words:", {key: value["summary"] for key, value in raw.items()})
    print("Hard decisions:", len(hard), "erasures", sum(r["observed_equal"] is None for r in hard),
          "disagreements", sum(r["observed_equal"] not in (None, r["decoded_equal"]) for r in hard))


if __name__ == "__main__":
    main()
