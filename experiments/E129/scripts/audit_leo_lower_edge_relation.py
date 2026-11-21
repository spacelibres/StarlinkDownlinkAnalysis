"""Test LEO's frozen three-position relation through exported and wideband signs."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_leo_early_symbol_roles import CODE5_LENGTH, CODE5_START, REFERENCE, bin_coordinates
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_reference


SOURCE = Path("reports/2026_09_28_sequence_semantics")
COORDINATES = ((3, 524), (3, 537), (5, 524))
PARITY = 1


def signed_bin(index):
    return (index + 512) % 1024 - 512


def resource_coordinates(coordinates):
    return [int(bin_coordinates([signed_bin(k)], symbol)[0]) for symbol, k in coordinates]


def score_sign_export(export, coordinates, parity):
    bins = export["bins"]
    frames = []
    selected = agreeing = 0
    for row in export["raw_signs"]:
        symbols = {entry["ofdm_symbol"]: entry for entry in row["symbols"]}
        if len(symbols) != len(row["symbols"]) or len(set(bins)) != len(bins):
            raise ValueError("Duplicate symbol or bin in LEO sign export")
        for entry in symbols.values():
            if any(len(entry[k]) != len(bins) or set(entry[k]) - {"0", "1"}
                   for k in ("rx0", "rx1", "selected")):
                raise ValueError("Invalid binary sign export")
            expected = "".join(a if keep == "1" and a == b else "?" for a, b, keep in
                               zip(entry["rx0"], entry["rx1"], entry["selected"], strict=True))
            if expected != entry["consensus"]:
                raise ValueError("Exported consensus disagrees with the receiver signs or mask")
            selected += entry["selected"].count("1")
            agreeing += len(bins) - expected.count("?")
        fields = {key: "".join(symbols[symbol][key][bins.index(k)] for symbol, k in coordinates)
                  for key in ("rx0", "rx1", "consensus")}
        frames.append({"frame": row["frame"], "rx0_candidate": fields["rx0"], "rx1_candidate": fields["rx1"],
                       "candidate_consensus": fields["consensus"]})
    if ([r["frame"] for r in frames] != export["evaluation_frames"]
            or len({r["frame"] for r in frames}) != len(frames)
            or selected != export["matched"]["count"] or agreeing / selected != export["matched"]["agreement"]):
        raise ValueError("Exported frame list or aggregate sign agreement changed")
    complete = [r["candidate_consensus"] for r in frames if "?" not in r["candidate_consensus"]]
    return {"expected_soft_sha256": export.get("sha256"), "frames": frames, "selected_signs": selected,
            "agreeing_selected_signs": agreeing, "complete_consensus_frames": len(complete),
            "consensus_matches": sum(bits.count("1") % 2 == parity for bits in complete),
            "unmasked_receiver_matches": [sum(r[f"rx{rx}_candidate"].count("1") % 2 == parity
                                             for r in frames) for rx in range(2)]}


def score_archive(artifact, rotation, resources):
    headers = read(artifact("v034-qin-header-survey", "E036", "qin_header_survey.json"))["frames"]
    by_frame = {r["frame"]: r for r in headers}
    run = "v032-qin-snapshot"
    snapshot = read(artifact(run, "E034", "qin_snapshot_source.json"))
    symbols, bins = np.array(COORDINATES).T
    rows = []
    for group in snapshot["hard_groups"]:
        with np.load(artifact(run, "E034", f"qin_hard_{group}.npz"), allow_pickle=False) as z:
            for frame, values in zip(z["frame"], z["symbols"], strict=True):
                r = by_frame[int(frame)]
                h1 = r["h1"]
                entries = h1["layout20"]["entries"]
                if (not h1["crc_ok"] or not r["h2"]["crc_ok"] or h1["needed_resources"] != CODE5_START
                        or entries[0]["mcs_candidate"] != 5 or entries[0]["quantity"] < 2
                        or r["h2_start"] <= max(resources)):
                    raise ValueError(f"Unexpected Code 5 allocation in Qin frame {frame}")
                signs = values[symbols - 1, bins] * rotation
                qualified = (abs(signs.imag) < 0.05) & (abs(abs(signs.real) - 1) < 0.05)
                bits = "".join(str(int(x.real >= 0)) if keep else "?" for x, keep in zip(signs, qualified, strict=True))
                rows.append({"frame": int(frame), "source_block": group, "bits": bits,
                             "matches": bits.count("1") % 2 == PARITY if qualified.all() else None})
    if len({r["frame"] for r in rows}) != len(rows):
        raise ValueError("Duplicate Qin frames in the hard snapshot")
    return rows


def score_raw_words(artifact, ratios, resources):
    run = "v010-code5-parity-02"
    metadata = read(artifact(run, "E012", "code5_parity.json"))["words"]
    with np.load(artifact(run, "E012", "code5_observations.npz"), allow_pickle=False) as z:
        bits, generator = z["hard_bits"], z["parity_generator"]
    half = CODE5_LENGTH // 2
    by_frame = {}
    for row, word in zip(metadata, bits, strict=True):
        if row["field"] != "first" or row["word"] not in (0, 1):
            continue
        if (row["start_resource"] != CODE5_START + row["word"] * CODE5_LENGTH
                or np.any((word[:half] @ generator) % 2 != word[half:])):
            raise ValueError("Wideband Code 5 row does not pass its frozen checks")
        words = by_frame.setdefault(row["frame"], {})
        if row["word"] in words:
            raise ValueError("Duplicate Code 5 word in independent wideband capture")
        words[row["word"]] = word
    word_offsets = [divmod(index - CODE5_START, CODE5_LENGTH) for index in resources]
    rows = []
    for frame, words in sorted(by_frame.items()):
        if not all(word in words for word, _ in word_offsets):
            continue
        values = np.array([int(words[word][offset]) for word, offset in word_offsets])
        signs = (1 - 2 * values) * ratios.real
        pattern = "".join(str(int(x >= 0)) for x in signs)
        rows.append({"frame": frame, "bits": pattern, "matches": pattern.count("1") % 2 == PARITY})
    return rows


def main():
    require_run_context()
    inputs, external = {}, {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    def leo(relative):
        payload = (EXTERNAL_LEO_ROOT / relative).read_bytes()
        external[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    local = SOURCE / "local"
    hypothesis = [r for r in leo(local / "lower_edge_parity.json")["rows"] if "local_transfers" in r]
    if len(hypothesis) != 1 or hypothesis[0]["coordinates"] != [list(x) for x in COORDINATES] or hypothesis[0]["parity"] != PARITY:
        raise ValueError("Original LEO hypothesis changed")
    middle = next(r for r in leo(local / "local_header_recovery.json")["rows"] if r["signal"] == "DS9-middle")
    last = leo(local / "ds9_last_header_audit.json")
    ds9 = {r["signal"]: score_sign_export(r, COORDINATES, PARITY) for r in (middle, last)}
    resources = resource_coordinates(COORDINATES)
    symbols, bins = np.array(COORDINATES).T
    inputs[REFERENCE] = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    template = loadmat(REFERENCE)["referenceTemplateRotations"]
    rotation = np.exp(-0.5j * np.pi * template[bins, symbols - 1])
    ratios = rotation * np.array([pn_reference(signed_bin(k), np.array([s]))[0] for s, k in COORDINATES])
    if np.any(abs(ratios.imag) > 1e-10):
        raise ValueError("Phase conversion is not a real sign")
    archive = score_archive(artifact, rotation, resources)
    independent = score_raw_words(artifact, ratios, resources)
    result = {"schema": "starlink-leo-lower-edge-relation/v1", "hypothesis": hypothesis[0],
              "leo_inputs_sha256": external,
              "project_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): h for p, h in inputs.items()},
              "resources": resources, "code5_word_and_offset": [divmod(r - CODE5_START, CODE5_LENGTH) for r in resources],
              "analytical_to_qin_real_signs": ratios.real.round().astype(int).tolist(),
              "ds9": ds9, "archive": archive, "raw_480msps": independent}
    destination = experiment_output("E129")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "leo_lower_edge_relation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    for name, rows in (("Qin <78", [r for r in archive if r["frame"] < 78]),
                       ("Qin >=78", [r for r in archive if r["frame"] >= 78]), ("480 MS/s", independent)):
        print(f"{name}: {sum(r['matches'] is True for r in rows)}/{sum(r['matches'] is not None for r in rows)}; "
              f"erasures {sum(r['matches'] is None for r in rows)}")
    for name, receipt in ds9.items():
        print(f"{name}: {receipt['agreeing_selected_signs']}/{receipt['selected_signs']} agreeing signs; "
              f"{receipt['complete_consensus_frames']}/{len(receipt['frames'])} complete candidate consensus")


if __name__ == "__main__":
    main()
