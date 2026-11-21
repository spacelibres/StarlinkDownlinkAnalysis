"""Recover and replay the shared high-parity core of four lower-rate formats."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH, paired_syndromes
from audit_cross_rate_checks import endpoint_words, record, shift_systematic
from audit_cross32_partial_decode import raw_inputs
from audit_qam_pn15 import equality_predictions
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_full_code import SYSTEMATIC as PARITY_START
from recover_mcs66_systematic_checks import LIFT, packed_arrays
from sparse_code5_core import binary_rank
from validate_mcs66_full_code import decode_words, invert_binary, remodulate


TARGETS = (71, 75, 78, 81)
MASK_SEED = 20261121132


def right_aligned(source, edge):
    mapped = np.where(source < PARITY_START, source + edge - PARITY_START, source)
    return np.where(mapped < 0, BIT_LENGTH, mapped)


def transmitted_system_rows(support, edge):
    return [row[row < edge] for row in support]


def virtual_zero(values, value):
    return np.column_stack((values, np.full(len(values), value, dtype=values.dtype)))


def compare_archived(cohorts, correct, shifted, left):
    result = {}
    arms = (correct.reshape(-1, LIFT, 14), shifted.reshape(-1, LIFT, 14), left.reshape(-1, LIFT, 14))
    for name, bits, seen, groups in cohorts:
        if not len(bits):
            continue
        b, m = virtual_zero(bits, False), virtual_zero(seen, True)
        items = []
        for reference, wrong, wrong_edge in zip(*arms, strict=True):
            matched, control = paired_syndromes(b, m, groups, reference, wrong)
            matched_edge, control_edge = paired_syndromes(b, m, groups, reference, wrong_edge)
            items.append({"right_aligned": matched, "shifted_systematic": control,
                          "right_aligned_against_left": matched_edge, "left_aligned": control_edge})
        result[name] = {"words": len(bits), "groups": np.unique(groups).tolist(), "families": items}
    return result


def replay_measured(code, support, edge, inverse, z, labels):
    frames, prior, hard, observed_pn = raw_inputs(code, z, labels, edge)
    points = np.array([complex(a["i"], a["q"]) for a in labels["cross32_mapping"]]) / np.sqrt(20)
    mapping = labels_as_bits(np.array([a["label"] for a in labels["cross32_mapping"]]), 5)
    parity_symbols = np.arange((PARITY_START + 4) // 5, BIT_LENGTH // 5)
    masked = np.sort(np.random.default_rng(MASK_SEED).choice(parity_symbols, len(parity_symbols) // 16, replace=False))
    coords = (masked[:, None] * 5 + np.arange(5)).ravel()
    prior[:, edge:PARITY_START] = 0.
    withheld = prior.copy()
    withheld[:, coords] = 0.
    ordinary, ordinary_rounds = decode_words(virtual_zero(prior, np.inf), support)
    recovered, masked_rounds = decode_words(virtual_zero(withheld, np.inf), support)
    core = np.r_[np.arange(edge), np.arange(PARITY_START, BIT_LENGTH)]
    if not np.all(ordinary_rounds) or not np.all(masked_rounds) or not np.array_equal(ordinary[:, core], recovered[:, core]):
        raise ValueError(f"MCS {code} did not preserve the core in both paths")
    system = recovered[:, :edge]
    rhs = np.column_stack([np.bitwise_xor.reduce(system[:, row], axis=1) for row in transmitted_system_rows(support, edge)])
    rhs = [int.from_bytes(row.tobytes(), 'little') for row in np.packbits(rhs, axis=1, bitorder='little')]
    parity = np.asarray([[(row & value).bit_count() % 2 for row in inverse] for value in rhs], dtype=bool)
    if not np.array_equal(parity, recovered[:, PARITY_START:BIT_LENGTH]):
        raise ValueError("Independent core encoder returned different parity bits")

    visible = recovered[:, :BIT_LENGTH].copy()
    visible[:, edge:PARITY_START] = False
    wrong = visible.copy()
    wrong[:, PARITY_START:] = np.roll(parity.reshape(len(parity), -1, LIFT), 1, axis=-1).reshape(parity.shape)
    y = z[f"mcs{code}_symbols"][:, masked]
    power = (abs(y) ** 2).sum(axis=1)
    error = (abs(remodulate(visible, points, mapping)[:, masked] - y) ** 2).sum(axis=1) / power
    wrong_error = (abs(remodulate(wrong, points, mapping)[:, masked] - y) ** 2).sum(axis=1) / power
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    after = ((visible != pn) & observed_pn).sum(axis=1)
    before = ((hard != pn) & observed_pn).sum(axis=1)
    if np.any(after) or np.any(error >= wrong_error):
        raise ValueError(f"MCS {code} did not preserve its received validation evidence")
    coverage = np.isin(np.arange(BIT_LENGTH), core)
    report = {"masked_parity_symbols": masked.tolist(), "unresolved_interval": [edge, PARITY_START],
              "rows": [{"frame": int(frame), "receiver_reserved": bool(frame in labels["frame_split"]["reserved"]),
                        "ordinary_rounds": int(ordinary_rounds[i]), "masked_rounds": int(masked_rounds[i]),
                        "pn_checked": int(observed_pn[i].sum()), "pn_hard_errors": int(before[i]),
                        "pn_decoded_errors": int(after[i]), "withheld_nmse": float(error[i]),
                        "wrong_parity_nmse": float(wrong_error[i])} for i, frame in enumerate(frames)]}
    return report, np.packbits(visible, axis=1), np.packbits(coverage)


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    layout = read(artifact("v068-pn-padding-starts", "E070", "pn_padding_starts.json"))
    with np.load(artifact("v126-mcs66-full-code", "E128", "mcs66_full_checks.npz"), allow_pickle=False) as z:
        source = z["row_support"]
        base_matrix = np.unpackbits(z["check_matrix_packed"], axis=1)
    parity = base_matrix[:, PARITY_START:]
    inverse = invert_binary(parity)
    shifted_source = shift_systematic(source, PARITY_START)
    run = "v125-cross-rate-checks"
    data = read(artifact(run, "E127", "cross_rate_checks.json"))
    split_exclusions = data["archive_excluded_groups"]
    middle = {}
    with np.load(artifact(run, "E127", "cross_rate_middle_words.npz"), allow_pickle=False) as z:
        for code in TARGETS:
            bits, seen = packed_arrays(z, (f"mcs{code}_bits", f"mcs{code}_seen"))
            middle[code] = record(bits, seen, data["archive_middle_words"][str(code)], "archive_middle", split_exclusions)
    run = "v120-cross-mcs-checks"
    endpoints = read(artifact(run, "E122", "cross_mcs_checks.json"))
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    raw = artifact("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    results, models, verified = {}, {}, {}
    with np.load(artifact(run, "E122", "cross_mcs_observations.npz"), allow_pickle=False) as z, np.load(raw, allow_pickle=False) as observations:
        for code in TARGETS:
            edge = layout["marker_systematic_ends"][str(code)]
            support = right_aligned(source, edge)
            shifted = right_aligned(shifted_source, edge)
            left = np.where((source >= edge) & (source < PARITY_START), BIT_LENGTH, source)
            matrix = np.zeros((len(support), BIT_LENGTH + 1), dtype=np.uint8)
            matrix[np.arange(len(support))[:, None], support] = 1
            rank = binary_rank(matrix[:, :BIT_LENGTH])
            if rank != BIT_LENGTH - PARITY_START or not np.array_equal(matrix[:, PARITY_START:BIT_LENGTH], parity):
                raise ValueError("Lower-rate core does not inherit a full high-parity matrix")
            ranges = list(endpoint_words(z, endpoints, code, split_exclusions)) + [middle[code]]
            tails = compare_archived(ranges, support, shifted, left)
            for check in tails["archive_terminal"]["families"]:
                if check["right_aligned"]["group_accuracy"] - check["shifted_systematic"]["group_accuracy"] < .08:
                    raise ValueError(f"Insufficient archived holdout for MCS {code}")
            record_data = {"systematic_end": edge, "rank": rank, "unresolved_parity_interval": [edge, PARITY_START],
                           "archive": tails}
            if code in (71, 78):
                received, bits, seen = replay_measured(code, support, edge, inverse, observations, labels)
                record_data["waveform"] = received
                verified[f"mcs{code}_bits"] = bits
                verified[f"mcs{code}_seen"] = seen
            results[code] = record_data
            models[f"mcs{code}_matrix"] = np.packbits(matrix[:, :BIT_LENGTH], axis=1)
            print(code, rank, "shared checks; unresolved parity bits", PARITY_START - edge,
                  "measured core words", len(record_data.get("waveform", {}).get("rows", [])), flush=True)
    out = experiment_output("E132")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "lower_rate_core_matrices.npz", **models)
    np.savez_compressed(out / "lower_rate_core_measured_words.npz", **verified)
    result = {"schema": "starlink-lower-rate-shared-core/v1", "mask_seed": MASK_SEED, "start_of_shared_parity": PARITY_START,
              "excluded_overlapping_archive_groups": split_exclusions, "results": results,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    (out / "lower_rate_core.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
