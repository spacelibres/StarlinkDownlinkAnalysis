"""Test frozen QC relations on MCS 66/70 before attempting to assign a new code."""

from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code28_reencoding import fft_binary_parities
from audit_ut_binary_tail import sha256
from code21_ring import binary_blocks, evaluate, pack_blocks
from code68_mapping_core import nullspace
from qc_figure_core import generator_from_shifts, qc_syndrome

EXTRACT = ROOT / "runs/v026-ut-qam-extract/artifacts/experiments/E028/results"
RECEIPT = ROOT / "runs/v026-ut-qam-extract/manifest.json"
TABLE = ROOT / "experiments/E018/figure12_shifts.json"
CODE21 = ROOT / "runs/v014-code21-repeat/artifacts/experiments/E016/results/code21_observations.npz"
SEED = 2026112129
LANES = list(permutations(range(4)))
COMPLEMENTS = ((np.arange(16)[:, None] >> np.arange(4)[None, :]) & 1).astype(np.uint8)


def slice_labels(observations, points, mapping):
    labels = []
    margins = []
    for batch in np.array_split(observations, max(1, (len(observations) + 23) // 24)):
        distance = abs(batch[..., None] - points) ** 2
        labels.append(distance.argmin(axis=-1).astype(np.uint8))
        two = np.partition(distance, 1, axis=-1)[..., :2]
        margins.append(two[..., 1] - two[..., 0])
    return mapping[np.concatenate(labels)], np.concatenate(margins)


def check_counts_by_lane(bits, shifts, lift):
    values = qc_syndrome(bits.reshape(-1, lift * shifts.shape[1]), shifts, lift)
    stride = values.reshape(len(values), len(shifts), lift // 4, 4)
    return stride.sum(axis=(0, 2)), len(values) * (lift // 4)


def complement_masks(shifts):
    """DC syndrome for one four-lane complement pattern, modulo the lift."""
    return np.stack([
        np.bitwise_xor.reduce(COMPLEMENTS[:, (np.arange(4)[None, :] + row[row >= 0][:, None]) % 4], axis=1)
        for row in shifts
    ], axis=1)


def high_margin_checks(bits, margin, shifts, lift):
    width = lift * shifts.shape[1]
    labels = bits.reshape(-1, width)
    reliable_bits = np.repeat(margin >= .08, 4, axis=1).reshape(len(labels), shifts.shape[1], lift)
    retained = np.hstack([
        np.logical_and.reduce([np.roll(reliable_bits[:, j, :], -int(d), axis=1)
                               for j, d in enumerate(row) if d >= 0])
        for row in shifts
    ])
    return qc_syndrome(labels, shifts, lift), retained


def frozen_mapping_search(symbol_bits, margins, training, shifts, lift):
    dc = complement_masks(shifts)
    variants = []
    for offset in (-1, 0, 1):
        bits = symbol_bits[:, 1 + offset:symbol_bits.shape[1] - 1 + offset]
        for index, lane in enumerate(LANES):
            arranged = bits[:, :, lane].reshape(len(bits), -1)
            c, total = check_counts_by_lane(arranged[training], shifts, lift)
            errors = np.where(dc != 0, total - c, c).sum(axis=(1, 2))
            n = total * c.size
            for number, e in enumerate(errors):
                variants.append((int(e), offset, index, number, n))
    errors, offset, lane, complement, total = min(variants)
    chosen = symbol_bits[:, 1 + offset:symbol_bits.shape[1] - 1 + offset, :][:, :, LANES[lane]]
    chosen = (chosen ^ COMPLEMENTS[complement]).reshape(len(chosen), -1)
    sliced_margins = margins[:, 1 + offset:margins.shape[1] - 1 + offset]
    held, strong = high_margin_checks(chosen[~training], sliced_margins[~training], shifts, lift)
    fixed = symbol_bits[:, 1:-1].reshape(len(symbol_bits), -1)
    fixed_syndrome = qc_syndrome(fixed.reshape(-1, lift * shifts.shape[1]), shifts, lift)
    return {
        "choices": len(variants), "selected_symbol_offset": offset,
        "selected_bit_lane_order": list(LANES[lane]), "selected_lane_complements": COMPLEMENTS[complement].tolist(),
        "selected_train_failed_checks": errors, "selected_train_total_checks": total,
        "frozen_gray_full_mean_failure": float(fixed_syndrome.mean()),
        "frozen_gray_full_zero_syndromes": int(np.count_nonzero(~fixed_syndrome.any(axis=1))),
        "held_failed_checks_per_word": held.sum(axis=1).tolist(),
        "held_checks_per_word": held.shape[1], "held_zero_syndromes": int(np.count_nonzero(~held.any(axis=1))),
        "held_high_margin_checks": int(strong.sum()),
        "held_high_margin_failed_fraction": float(held[strong].mean()),
    }


def code21_fixed(bits, p21):
    b = bits.reshape(-1, 8192)
    parity, rounding = fft_binary_parities(b[:, :5120].reshape(-1, 10, 512), p21)
    failures = np.count_nonzero(parity != b[:, 5120:], axis=1)
    return {
        "tested_candidate_halfwords": len(b),
        "mean_mismatched_parity": float(failures.mean() / 3072),
        "zero_disagreement_halfwords": int(np.count_nonzero(failures == 0)),
        "fft_maximum_rounding_residual": rounding,
    }


def block_parity_rank(bits, training):
    result = []
    for lift in (256, 512, 1024, 2048, 4096):
        reduced = (bits.reshape(len(bits), -1, lift).sum(axis=2) % 2).astype(np.uint8)
        packed = [int.from_bytes(v, "little") for v in np.packbits(reduced, axis=1, bitorder="little")]
        width = reduced.shape[1]
        kern, train_rank = nullspace([v for v, use in zip(packed, training, strict=True) if use], width)
        _, combined_rank = nullspace(packed, width)
        result.append({"lift": lift, "projected_coordinates": width, "train_rank": train_rank,
                       "train_nullity": len(kern), "combined_rank": combined_rank,
                       "validated_nullity": width - combined_rank})
    return result


def synthetic_control(points, mapping, shifts, lift):
    rng = np.random.default_rng(SEED + lift)
    n = 16
    k = (shifts.shape[1] - shifts.shape[0]) * lift
    data = rng.integers(0, 2, (n, k), dtype=np.uint8)
    parity = binary_blocks(evaluate(pack_blocks(data, lift), generator_from_shifts(shifts, lift), lift), lift)
    words = np.column_stack((data, parity))
    symbol_indices = (words.reshape(n, -1, 4)[:, :, None, :] == mapping[None, None, :, :]).all(axis=-1).argmax(axis=-1)
    values = points[symbol_indices]
    response = []
    for variance in (0., .032, .06):
        noise = np.sqrt(variance / 2) * (rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape))
        received, margin = slice_labels(values + noise, points, mapping)
        b = received.reshape(n, -1)
        synd, usable = high_margin_checks(b, margin, shifts, lift)
        response.append({"complex_variance": variance, "symbol_decision_fraction_wrong": float(np.any(b.reshape(n, -1, 4) != words.reshape(n, -1, 4), axis=2).mean()),
                         "zero_syndromes": int(np.count_nonzero(~synd.any(axis=1))), "failed_check_fraction": float(synd.mean()),
                         "strong_check_count": int(usable.sum()), "strong_check_failed_fraction": float(synd[usable].mean())})
    return response


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    require_run_context()
    meta = EXTRACT / "ut_qam_extraction.json"
    data = EXTRACT / "ut_qam_symbols.npz"
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in (meta, data, RECEIPT, TABLE, CODE21)}
    artifacts = read(RECEIPT)["changed_artifacts"]
    for path in (meta, data):
        if hashes[path.relative_to(ROOT).as_posix()] != artifacts[f"experiments/E028/results/{path.name}"]["sha256"]:
            raise ValueError("Frozen QAM observation artifact changed")
    manifest = read(meta)
    tables = read(TABLE)
    shifts = {key: np.array(tables[str(key)]) for key in (512, 1024)}
    with np.load(CODE21, allow_pickle=False) as archive:
        p21 = archive["parity_generator"][::512].reshape(10, 6, 512)
    reports = {}
    with np.load(data, allow_pickle=False) as archive:
        points, mapping = archive["constellation_points"], archive["frozen_gray_mapping"]
        for code in (66, 70):
            group = archive[f"mcs{code}_frames"]
            training = np.isin(group, manifest["calibration_frames"])
            refinements = {}
            for key in ("baseline", "refined"):
                bits, margin = slice_labels(archive[f"mcs{code}_{key}"], points, mapping)
                fixed = bits[:, 1:-1].reshape(len(bits), -1)
                refinements[key] = {
                    "mean_gray_ones_fraction": float(fixed.mean()),
                    "symbol_margin_quantiles": np.quantile(margin[:, 1:-1], [.01, .05, .5]).tolist(),
                    "published_code39": frozen_mapping_search(bits, margin, training, shifts[1024], 1024),
                    "two_halfwords_code3": frozen_mapping_search(bits, margin, training, shifts[512], 512),
                    "two_halfwords_code21_fixed_gray": code21_fixed(fixed, p21),
                    "coarse_parity_projection": block_parity_rank(fixed, training),
                }
            reports[str(code)] = {"word_count": len(group), "reserved_word_count": int(np.count_nonzero(~training)),
                                  "front_end_variants": refinements}
            print("MCS", code, len(group), "words", flush=True)
        controls = {str(lift): synthetic_control(points, mapping, s, lift) for lift, s in shifts.items()}
    out = experiment_output("E028")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ut_qam_fec_audit.json").write_text(json.dumps({
        "schema": "starlink-ut-qam-fec-transfer/v1", "input_hashes": hashes, "seed": SEED,
        "frames_split": {"calibration": manifest["calibration_frames"], "reserved": manifest["reserved_frames"]},
        "results": reports, "synthetic_known_code_controls": controls,
        "scope": "Finite Gray lane, complement and symbol-offset transfer, plus coarse hard-block parity signatures. "
                 "Exact membership and observed bit errors are distinct, and no unknown encoder is fitted.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
