"""Certify what LEO frequency windows can determine from the complete MCS 66/85 code."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context
from audit_code5_directory_records import pilot_archive_windows
from audit_leo_code5_projection import nominal_capture_windows, selected_windows
from audit_leo_recording_coverage import CARRIER_SPACING_HZ, signed_bin
from audit_mcs66_partial_decoder import GUARD_BITS
from audit_ut_repeated_data import checked_artifact, read
from layout_core import NONPILOT
from recover_mcs66_full_code import SYSTEMATIC


LENGTH = 16384
CHECKS = LENGTH - SYSTEMATIC
GROUP_SIZE = 64
FORMATS = ((66, 4), (85, 5))


def observed_bits(mask, origin, width, length=LENGTH):
    return mask[(np.asarray(origin)[..., None] + np.arange(length) // width) % len(mask)]


def column_basis(columns, indices, rank_limit):
    pivots = {}
    basis = []
    for index in indices:
        value = columns[index]
        while value:
            pivot = value.bit_length()
            if pivot not in pivots:
                pivots[pivot] = value
                basis.append(int(index))
                break
            value ^= pivots[pivot]
        if len(basis) == rank_limit:
            break
    return basis


def disjoint_bases(columns, hidden, rank_limit):
    descending = np.flatnonzero(hidden)[::-1].tolist()
    even = [i for i in descending if i % 2 == 0]
    odd = [i for i in descending if i % 2 != 0]
    first = column_basis(columns, even + odd, rank_limit)
    used = set(first)
    second = column_basis(columns, (i for i in odd + even if i not in used), rank_limit)
    return first, second


def residual_dimension(columns, known, rank_limit):
    hidden = np.flatnonzero(~known)
    rank = len(column_basis(columns, hidden[::-1], rank_limit))
    return len(hidden) - rank, rank


def leo_windows(artifact, external):
    catalogue = read(artifact("v069-leo-recording-coverage", "E071", "leo_recording_coverage.json"))
    pilot = read(artifact("v097-leo-pilot-archive", "E099", "leo_pilot_archive.json"))
    windows = (selected_windows(external) + nominal_capture_windows(catalogue)
               + list(pilot_archive_windows(pilot, external)))
    native = read(artifact("v105-leo-native-clock-coverage-final", "E107", "leo_native_clock_coverage.json"))
    edge = catalogue["separate_native_glrt_capture"]
    if native["capture"] != edge["capture"] or native["grid"]["nominal_511_baseband_hz"] != edge["target_baseband_hz"]:
        raise ValueError("Inconsistent native 0181 geometry")
    center = edge["target_baseband_hz"] + (NONPILOT - 511) * CARRIER_SPACING_HZ
    limit = min(edge["sample_rate_hz"], edge["rf_bandwidth_hz"]) / 2
    for row in native["grid"]["rows"]:
        for method, value in row["frequencies"].items():
            carriers = NONPILOT[abs(center + value["shift_hz"]) <= limit]
            windows.append({"kind": "native_cfo_geometry", "carriers": carriers.tolist(),
                            "labels": [f'{edge["capture"]}/{row["mode"]}/{method}']})

    relative = Path("reports/2026_09_27_ds7_header/local/holdout-upper/recurring-words.json")
    ds7 = (EXTERNAL_LEO_ROOT / relative).read_bytes()
    external[relative.as_posix()] = sha256(ds7).hexdigest()
    windows.append({"kind": "10msps_exported_data_bins", "labels": ["DS7 upper edge"],
                    "carriers": [signed_bin(k) for k in json.loads(ds7)["subcarrier_indices"]]})
    e129 = read(artifact("v127-leo-lower-edge", "E129", "leo_lower_edge_relation.json"))
    for filename in ("local_header_recovery.json", "ds9_last_header_audit.json"):
        relative = Path("reports/2026_09_28_sequence_semantics/local") / filename
        payload = (EXTERNAL_LEO_ROOT / relative).read_bytes()
        external[relative.as_posix()] = sha256(payload).hexdigest()
        if external[relative.as_posix()] != e129["leo_inputs_sha256"][relative.as_posix()]:
            raise ValueError("DS9 sign export differs from E129")
        receipt = json.loads(payload)
        if "rows" in receipt:
            receipt = next(row for row in receipt["rows"] if row["signal"] == "DS9-middle")
        windows.append({"kind": "10msps_exported_data_bins", "labels": [receipt["signal"]],
                        "carriers": [signed_bin(k) for k in receipt["bins"]]})
    if any(set(row["carriers"]) - set(NONPILOT) for row in windows):
        raise ValueError("Window includes an invalid non-pilot resource")
    return windows


def conditional_words(artifact):
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    retained = set(labels["frame_split"]["reserved"])
    mcs66 = read(artifact("v086-mcs66-decoder", "E088", "mcs66_partial_decoder.json"))["raw_iq"]["word_observations"]
    for row in mcs66:
        if row["frame"] in retained:
            yield (66, 4, row["frame"], row["resource_start"], row["bound"] + GUARD_BITS, SYSTEMATIC)
    extraction = read(artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_extraction.json"))
    allocation_by_frame = {r["frame"]: r for r in extraction["regions"] if r["format"] == 85}
    for row in labels["word_results"]:
        if row["format"] == 85 and row["accepted"] and row["allocation_final_word"] and row["frame"] in retained:
            allocation = allocation_by_frame[row["frame"]]
            yield (85, 5, row["frame"], allocation["stop"] - (LENGTH + 4) // 5,
                   row["start"] + GUARD_BITS, min(row["stop"] - GUARD_BITS, SYSTEMATIC))


def main():
    require_run_context()
    dependencies, external = {}, {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, dependencies)

    with np.load(artifact("v126-mcs66-full-code", "E128", "mcs66_full_checks.npz"), allow_pickle=False) as data:
        support = data["row_support"]
        matrix = np.unpackbits(data["check_matrix_packed"], axis=1)
    if support.shape != (CHECKS, 14) or matrix.shape != (CHECKS, LENGTH):
        raise ValueError("Unexpected frozen check matrix shape")
    if not np.all(matrix.sum(axis=1) == support.shape[1]) or not np.array_equal(np.nonzero(matrix)[1], support.ravel()):
        raise ValueError("Support does not describe frozen matrix")
    columns = [int.from_bytes(row.tobytes(), 'little') for row in np.packbits(matrix.T, axis=1, bitorder='little')]
    windows = leo_windows(artifact, external)
    nominal = [np.isin(NONPILOT, row["carriers"]) for row in windows if row["kind"] == "catalogued_nominal"]
    union = np.isin(NONPILOT, sorted({k for row in windows for k in row["carriers"]}))
    certificates, summary = {}, {}
    origins = np.arange(len(NONPILOT))
    for code, width in FORMATS:
        groups = []
        for start in range(0, len(origins), GROUP_SIZE):
            visible = observed_bits(union, origins[start:start + GROUP_SIZE], width)
            first, second = disjoint_bases(columns, ~visible.any(axis=0), CHECKS)
            if len(first) != CHECKS or len(second) != CHECKS:
                raise ValueError(f"Could not certify all origins in MCS {code} group {start}")
            groups.append([first, second])
        certificates[f"mcs{code}_basis_columns"] = np.asarray(groups, dtype=np.uint16)
        count = observed_bits(union, origins, width).sum(axis=1)
        summary[code] = {"origins": len(origins), "groups": len(groups),
                         "union_observed_bit_range": [int(count.min()), int(count.max())],
                         "union_ambiguity_dimension_range": [int(SYSTEMATIC - count.max()), int(SYSTEMATIC - count.min())]}
    control = np.ones(LENGTH, dtype=bool)
    control[0] = False
    if residual_dimension(columns, control, CHECKS) != (0, 1):
        raise ValueError("Single erased bit should be recoverable with all others known")

    conditional = []
    for code, width, frame, start, first, stop in conditional_words(artifact):
        if not 0 < first < stop <= SYSTEMATIC:
            raise ValueError("Invalid guarded PN interval")
        known_pn = (np.arange(LENGTH) >= first) & (np.arange(LENGTH) < stop)
        base, _ = residual_dimension(columns, known_pn, CHECKS)
        scenarios = {}
        for name, mask in [(f"nominal_{i}", mask) for i, mask in enumerate(nominal)] + [("all_windows_union", union)]:
            observed = observed_bits(mask, start, width)
            remaining, rank = residual_dimension(columns, observed | known_pn, CHECKS)
            direct = int((observed[:SYSTEMATIC] & ~known_pn[:SYSTEMATIC]).sum())
            scenarios[name] = {"remaining_dimension": remaining, "erasure_rank": rank,
                               "direct_unknown_systematic_bits": direct, "extra_linear_constraints": base - remaining - direct}
        conditional.append({"code": code, "frame": frame, "resource_start": start, "pn_interval": [first, stop],
                            "initial_dimension": base, "scenarios": scenarios})
    result = {"schema": "starlink-leo-full-code-observability/v1", "windows": windows,
              "union_carriers": NONPILOT[union].tolist(), "group_size": GROUP_SIZE,
              "width_results": summary, "conditional_terminal_words": conditional,
              "external_input_sha256": external,
              "artifact_input_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in dependencies.items()}}
    path = experiment_output("E130")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "leo_full_code_certificates.npz", **certificates)
    (path / "leo_full_code_observability.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(len(windows), "modeled windows,", int(union.sum()), "union carriers", summary)
    for row in conditional:
        print(row['code'], row['frame'], row['initial_dimension'],
              {name: value['remaining_dimension'] for name, value in row['scenarios'].items()})


if __name__ == "__main__":
    main()
