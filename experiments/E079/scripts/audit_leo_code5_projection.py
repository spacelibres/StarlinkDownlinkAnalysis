"""Test whether the LEO receiver windows identify the Code 5 count in an ideal channel."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_code5_frame_count import RELATIVE_BIT
from audit_h2_projection import GRID, NONPILOT, SELECTION_ROOT, visible_positions
from audit_leo_recording_coverage import carrier_if_hz
from audit_ut_repeated_data import checked_artifact, read
from sparse_code5_core import binary_rank


def field_rank(code_rows, target):
    return binary_rank(code_rows) - binary_rank(code_rows[:, ~target])


def selected_windows(source_hashes):
    windows = {}
    for dataset in ("DSX01-S05", "paired_RX"):
        local = SELECTION_ROOT / dataset
        for name in ("selection.json", "selected_iq.npz"):
            relative = (local / name).relative_to(ROOT / "data/raw/leo_tracker_snapshot")
            payload = (EXTERNAL_LEO_ROOT / relative).read_bytes()
            if payload != (local / name).read_bytes():
                raise ValueError(f"External input no longer matches the frozen snapshot: {relative}")
            source_hashes[relative.as_posix()] = sha256(payload).hexdigest()
        selection = read(local / "selection.json")
        with np.load(local / "selected_iq.npz", allow_pickle=False) as iq:
            for row in selection["windows"]:
                name = f"iq_{row['number']}"
                digest = sha256(iq[name].tobytes()).hexdigest()
                offset = row["winner"]["fractional_tracking_cfo_hz"]
                mask = abs((NONPILOT + 1024 - 531.5) * GRID + offset) <= selection["sample_rate_hz"] / 2
                carriers = NONPILOT[mask].tolist()
                window = windows.setdefault(digest, {"kind": "recorded_narrowband", "labels": [], "carriers": carriers})
                if window["carriers"] != carriers:
                    raise ValueError(f"Duplicate window has inconsistent carrier masks: {dataset}/{name}")
                window["labels"].append(f"{dataset}/{name}")
    return list(windows.values())


def nominal_capture_windows(catalogue):
    rows = catalogue["cohort"] + [catalogue["separate_native_glrt_capture"]]
    masks = {}
    for row in rows:
        limit = min(row["sample_rate_hz"], row["rf_bandwidth_hz"]) / 2
        frequencies = [carrier_if_hz(row["channel"], int(k)) - row["tuned_if_hz"] for k in NONPILOT]
        carriers = tuple(NONPILOT[abs(np.array(frequencies)) <= limit].tolist())
        mask = masks.setdefault(carriers, {"kind": "catalogued_nominal", "labels": [], "carriers": list(carriers)})
        mask["labels"].append(row["capture"])
    return list(masks.values())


def measure_window(window, offsets, code):
    positions = visible_positions(228, 2048, window["carriers"])
    # Give the receiver the Code 5 start, PN sequence and first byte (offset L).
    equations = code[list(positions), 8:]
    data_bits = np.arange(8, 1024)
    by_offset = {}
    for offset in offsets:
        start = 8 * offset + RELATIVE_BIT
        by_offset[str(offset)] = {
            str(width): field_rank(equations, (data_bits >= start) & (data_bits < start + width))
            for width in (14, 32)
        }
    return {**window, "codeword_coordinates": list(positions), "field_rank_by_offset": by_offset}


def main():
    require_run_context()
    inputs = {}
    source_hashes = {}
    captures = read(checked_artifact("v069-leo-recording-coverage", "E071", "leo_recording_coverage.json", inputs))
    counts = read(checked_artifact("v077-code5-time-links", "E079", "code5_frame_count.json", inputs))
    offsets = sorted({row["offset_bytes"] for rows in counts["rows"].values() for row in rows})
    generator = checked_artifact("v010-code5-parity-02", "E012", "code5_observations.npz", inputs)
    with np.load(generator, allow_pickle=False) as data:
        code = np.r_[np.eye(1024, dtype=np.uint8), data["parity_generator"].T]
    windows = selected_windows(source_hashes) + nominal_capture_windows(captures)
    result = {
        "schema": "starlink-leo-code5-projection/v1",
        "external_inputs_sha256": source_hashes,
        "artifact_inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
        "known_inputs": ["Code 5 begins at resource 228", "fixed scrambling convention", "pointer byte L"],
        "offsets_tested": offsets,
        "windows": [measure_window(window, offsets, code) for window in windows],
    }
    out = experiment_output("E079")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leo_code5_projection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["windows"]:
        maxima = {width: max(v[width] for v in row["field_rank_by_offset"].values()) for width in ("14", "32")}
        print(row["kind"], row["labels"][0], len(row["labels"]), len(row["codeword_coordinates"]), maxima)


if __name__ == "__main__":
    main()
