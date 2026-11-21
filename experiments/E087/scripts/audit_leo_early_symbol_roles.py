"""Locate LEO's 24 early-symbol decisions in the frozen wideband allocation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from receiver_models import pn_reference


CODE5_START = 228
CODE5_LENGTH = 2048
REFERENCE = PROJECT_ROOT / "references/qin/referenceTemplate.mat"
LEO_REPORTS = EXTERNAL_LEO_ROOT / "reports"
DS7 = LEO_REPORTS / "2026_09_27_ds7_header/local"
UT = LEO_REPORTS / "2026_09_27_ut_header/local/fullband"
OFFSETS = (-2, -1, 0, 1, 2)


def distance(values, target):
    if not np.allclose(values.imag, 0, atol=1e-10):
        return None
    count = int(np.count_nonzero((values.real > 0) != target))
    return {"same_polarity": count, "opposite_polarity": len(target) - count}


def bin_coordinates(bins, symbol):
    by_bin = {int(k): i for i, k in enumerate(NONPILOT)}
    positions = np.array([by_bin[k] for k in bins])
    return (symbol - 2) * len(NONPILOT) + positions


def main():
    require_run_context()
    inputs, external = {}, {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    def leo(path):
        content = path.read_bytes()
        external[path.relative_to(EXTERNAL_LEO_ROOT).as_posix()] = hashlib.sha256(content).hexdigest()
        return json.loads(content)

    source = leo(DS7 / "ut-ds7-header-comparison.json")
    ds7_path = DS7 / "holdout-upper/recurring-words.json"
    ds7 = leo(ds7_path)
    if external[ds7_path.relative_to(EXTERNAL_LEO_ROOT).as_posix()] != source["ds7_source_sha256"]:
        raise ValueError("DS7 decisions differ from the historical comparison receipt")
    ds7_patterns = {r["mean_sign_pattern"] for phase in ds7["words"]
                    for r in phase["validation"] if r["source"] == "combined"}
    ut_patterns = {r["word"] for r in source["ut"]}
    if len(ut_patterns) != 1 or ds7_patterns != {source["ds7_pattern"]}:
        raise ValueError("Expected stable words in the recorded validation exports")
    ut_target = np.array(list(ut_patterns.pop())) == "1"
    ds7_target = np.array(list(source["ds7_pattern"])) == "1"
    bins = source["bins"]
    symbol = source["ofdm_symbol"]
    if bins != ds7["subcarrier_indices"] or symbol != ds7["ofdm_symbol"]:
        raise ValueError("The source exports do not share resource coordinates")
    resources = bin_coordinates(bins, symbol)
    word_numbers, word_offsets = np.divmod(resources - CODE5_START, CODE5_LENGTH)
    if len(set(word_numbers)) != 1 or max(word_offsets) >= CODE5_LENGTH // 2:
        raise ValueError("The probe coordinates leave one Code 5 systematic block")
    word_number = int(word_numbers[0])

    ut_meta = leo(UT / "results.json")
    run = "v022-ut-replay"
    rx = read(artifact(run, "E014", "ut_transfer.json"))
    reference_bytes = REFERENCE.read_bytes()
    inputs[REFERENCE] = hashlib.sha256(reference_bytes).hexdigest()
    for name, digest in ut_meta["input_sha256"].items():
        if name.endswith("exemplar250-257.bin") and digest != rx["source_sha256"]:
            raise ValueError("The UT waveform differs from the LEO source")
        if name.endswith("referenceTemplate.mat") and digest != inputs[REFERENCE]:
            raise ValueError("The Qin template differs from the LEO source")
    rotations = loadmat(REFERENCE)["referenceTemplateRotations"]
    q = np.exp(.5j * np.pi * rotations[bins, symbol - 1])
    pn = np.array([pn_reference(k, np.array([symbol]))[0] for k in bins])
    with np.load(artifact(run, "E014", "ut_observations.npz"), allow_pickle=False) as z:
        bits = z["hard_code5_words"]
    with np.load(artifact("v010-code5-parity-02", "E012", "code5_observations.npz"), allow_pickle=False) as z:
        generator = z["parity_generator"]
    captures = []
    by_candidate = {row["candidate_index"]: row for row in rx["candidates"]}
    for i, row in enumerate(rx["words"]):
        if row["word"] != word_number:
            continue
        candidate = by_candidate[row["candidate_index"]]
        if not candidate["h1"]["crc_ok"] or row["start_resource"] != CODE5_START + word_number * CODE5_LENGTH:
            raise ValueError("The wideband first allocation or word offset changed")
        end = int(candidate["predicted_h2_start"])
        if end <= max(resources):
            raise ValueError("The probe crosses the predicted H2 boundary")
        word = bits[i]
        half = CODE5_LENGTH // 2
        mismatch = int(np.count_nonzero((word[:half] @ generator) % 2 != word[half:]))
        shifts = {}
        for shift in OFFSETS:
            signs = 1 - 2 * word[word_offsets + shift].astype(float)
            shifts[shift] = distance(signs * pn * q.conj(), ut_target)
        captures.append({"candidate": row["candidate_index"], "h2_start": end, "code5_full_parity_errors": mismatch,
                         "ut_distances_by_offset": shifts})

    archive_run = "v034-qin-header-survey"
    archive = read(artifact(archive_run, "E036", "qin_header_survey.json"))
    boundaries = {r["frame"]: r["h2_start"] for r in archive["frames"]
                  if r["h1"]["crc_ok"] and r["h2"]["crc_ok"]}
    snapshot = "v032-qin-snapshot"
    source_index = read(artifact(snapshot, "E034", "qin_snapshot_source.json"))
    archive_rows = []
    for name in source_index["hard_groups"]:
        with np.load(artifact(snapshot, "E034", f"qin_hard_{name}.npz"), allow_pickle=False) as z:
            for frame, samples in zip(z["frame"], z["symbols"], strict=True):
                h2_start = boundaries[int(frame)]
                if max(resources) >= h2_start:
                    raise ValueError("The archive probe crosses the verified H2 boundary")
                values = samples[symbol - 1, bins] * q.conj()
                archive_rows.append({
                    "frame": int(frame), "h2_start": h2_start, "ds7_distance": distance(values, ds7_target),
                    "ut_distance": distance(values, ut_target),
                })
    distances = Counter(min(r["ds7_distance"].values()) for r in archive_rows if r["ds7_distance"] is not None)
    result = {
        "schema": "starlink-leo-early-symbol-roles/v1",
        "project_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
        "external_leo_inputs_sha256": external,
        "shared_ut_iq_sha256": rx["source_sha256"],
        "symbol": symbol, "bins": bins, "nonpilot_resources": resources.tolist(),
        "code5_word_zero_based": word_number, "zero_based_systematic_positions": word_offsets.tolist(),
        "ut_captures": captures, "archive_rows": archive_rows,
        "crc_valid_archive_headers": len(boundaries), "crc_valid_archive_min_h2_start": min(boundaries.values()),
        "archive_ds7_hamming_modulo_polarity": dict(sorted(distances.items())),
        "ds7_layout_confirmed": False,
    }
    out = experiment_output("E087")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leo_early_symbol_roles.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Resources {resources[0]}..{resources[-1]}, word {word_number}, "
          f"systematic bits {word_offsets[0]}..{word_offsets[-1]}; checked {len(captures)} UT words, "
          f"{len(boundaries)} header pairs, {len(archive_rows)} archive patterns, "
          f"closest DS7 archive distance {min(distances)}")


if __name__ == "__main__":
    main()
