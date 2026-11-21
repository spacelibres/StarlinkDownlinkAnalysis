"""Test block rotations of sparse relations on unregistered parity coordinates."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_gain_calibrated_folds import EXTRACTION
from audit_sparse_padding_checks import cohort
from audit_ut_repeated_data import checked_artifact, read


SPARSE = "v080-sparse-padding-final"
LIFTS = (128, 256, 512, 1024, 2048)


def canonical_support(positions, length):
    positions = np.asarray(positions)
    blocks, offsets = positions // length, positions % length
    return min(tuple(sorted((int(b), int(o)) for b, o in zip(blocks, (offsets - t) % length, strict=True)))
               for t in offsets[blocks == blocks.min()])


def orbit_positions(shape, length):
    support = np.array(shape, dtype=int)
    return support[:, 0] * length + (np.arange(length)[:, None] + support[:, 1]) % length


def project(bits, visible, positions):
    return np.bitwise_xor.reduce(bits[:, positions], axis=-1), visible[:, positions].all(axis=-1)


def summarize(parity, visible, reference, chosen, groups):
    mask = visible & chosen
    agree = (parity == reference) & mask
    rates = []
    for group in np.unique(groups):
        use = groups == group
        n = mask[use].sum()
        if n:
            rates.append(float(agree[use].sum() / n))
    return {"checks": int(mask.sum()), "matches": int(agree.sum()), "groups": len(rates),
            "equal_group_accuracy": sum(rates) / len(rates) if rates else None}


def predict_orbit(shape, length, nominated, padded, ordinary):
    positions = orbit_positions(shape, length)
    padded_bits, padded_visible, fit, groups = padded
    parity, visible = project(padded_bits, padded_visible, positions)
    group_votes = np.zeros(len(positions))
    group_support = np.zeros(len(positions), dtype=int)
    for group in np.unique(groups[fit]):
        use = fit & (groups == group)
        n = visible[use].sum(axis=0)
        k = (parity[use] & visible[use]).sum(axis=0)
        group_votes += (2 * k - n) / np.maximum(n, 1)
        group_support += n > 0
    reference = group_votes > 0
    confident = (group_support >= 8) & (abs(group_votes) / np.maximum(group_support, 1) >= .8)
    registered = np.array([tuple(sorted(c.tolist())) in nominated for c in positions])
    unseen = confident & ~registered
    data = parity[~fit], visible[~fit]
    control_bits, control_visible, control_groups = ordinary
    controls = project(control_bits, control_visible, positions)
    return {"length": length, "shape": shape, "registered_rotations": int(registered.sum()),
            "confident_signs": int(confident.sum()), "new_confident_signs": int(unseen.sum()),
            "fixed_affine_constants": [int(value) if enough else None
                                       for value, enough in zip(reference, confident, strict=True)],
            "held": summarize(*data, reference, confident, groups[~fit]),
            "held_unregistered_rotations": summarize(*data, reference, unseen, groups[~fit]),
            "held_reference_shift_one": summarize(*data, np.roll(reference, 1),
                                                   confident & np.roll(confident, 1), groups[~fit]),
            "held_nonterminal": summarize(*controls, reference, confident, control_groups)}


def audit_label(rows, control_rows, source, candidates, label):
    fit = np.array([r["calibration"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    padded = (*cohort(source, f"short_{label}", rows), fit, groups)
    use = np.array([not r["calibration"] for r in control_rows])
    bits, seen = cohort(source, f"nonterminal_{label}", control_rows)
    ordinary = bits[use], seen[use], np.array([r["group"] for r in control_rows])[use]
    outcomes = []
    for z in LIFTS:
        if padded[0].shape[1] % z:
            continue
        shapes = defaultdict(set)
        for row in candidates:
            support = tuple(row["coordinates"])
            shapes[canonical_support(support, z)].add(support)
        ordered = sorted(shapes, key=lambda shape: (-len(shapes[shape]), shape))
        outcomes.extend(predict_orbit(shape, z, shapes[shape], padded, ordinary) for shape in ordered[:2])
    return outcomes


def main():
    require_run_context()
    inputs = {}
    meta = read(checked_artifact(EXTRACTION, "E082", "gain_padded_words.json", inputs))["cohorts"]
    sparse = read(checked_artifact(SPARSE, "E082", "sparse_padding_checks.json", inputs))["formats"]
    data = checked_artifact(EXTRACTION, "E082", "gain_padded_words.npz", inputs)
    with np.load(data, allow_pickle=False) as z:
        outcomes = {str(label): audit_label(meta[f"short_{label}"], meta[f"nonterminal_{label}"], z,
                                            sparse[str(label)]["frozen_candidates"], label) for label in (66, 70)}
    result = {"schema": "starlink-sparse-qc-orbits/v1", "outcomes": outcomes,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E082")
    out.mkdir(parents=True, exist_ok=True)
    (out / "sparse_qc_orbits.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, arms in outcomes.items():
        for arm in arms:
            print(label, arm["length"], arm["shape"], arm["registered_rotations"], arm["held_unregistered_rotations"])


if __name__ == "__main__":
    main()
