"""Replay the most constrained terminal-word case with LEO visibility masks."""

from functools import reduce
import json
from operator import xor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross32_partial_decode import decode
from audit_leo_full_code_observability import LENGTH, NONPILOT, SYSTEMATIC, observed_bits
from audit_mcs66_partial_decoder import syndrome_count
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from validate_mcs66_full_code import inputs85


def solve_erasures(columns, bits, known):
    """Return a full word only when the supplied values imply a unique codeword."""
    known = np.asarray(known, dtype=bool)
    bits = np.asarray(bits, dtype=bool)
    hidden = np.flatnonzero(~known)
    pivots = {}
    for i, index in enumerate(hidden):
        value, coefficient = columns[index], 1 << i
        while value:
            pivot = value.bit_length()
            if pivot not in pivots:
                pivots[pivot] = value, coefficient
                break
            previous, lineage = pivots[pivot]
            value ^= previous
            coefficient ^= lineage
    target = reduce(xor, (columns[i] for i in np.flatnonzero(bits & known)), 0)
    recovered = 0
    while target:
        pivot = target.bit_length()
        if pivot not in pivots:
            return None, len(pivots), False
        previous, lineage = pivots[pivot]
        target ^= previous
        recovered ^= lineage
    if len(pivots) != len(hidden):
        return None, len(pivots), True
    out = bits.copy()
    out[hidden] = [(recovered >> i) & 1 for i in range(len(hidden))]
    return out, len(pivots), True


def main():
    require_run_context()
    inputs = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    geometry = read(artifact("v128-leo-full-code-observability", "E130", "leo_full_code_observability.json"))
    row = min(geometry["conditional_terminal_words"], key=lambda r: r["scenarios"]["all_windows_union"]["remaining_dimension"])
    if (row["code"], row["frame"]) != (85, 5):
        raise ValueError("The geometrically selected terminal word changed")
    labels = read(artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    witness = next(w for w in labels["word_results"] if w["format"] == 85 and w["frame"] == row["frame"] and w["accepted"])
    index = witness["candidate_index"]
    _, _, _, frames, prior, hard, _ = inputs85(artifact, labels)
    retained = np.flatnonzero(np.isin(frames, labels["frame_split"]["reserved"]))
    local = np.flatnonzero(retained == index).item()
    if frames[index] != row["frame"]:
        raise ValueError("Terminal index does not match its frozen frame")
    mask = np.isin(NONPILOT, geometry["union_carriers"])
    observed = observed_bits(mask, row["resource_start"], 5)
    pn = equality_predictions(np.arange(LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    with np.load(artifact("v126-mcs66-full-code", "E128", "mcs66_full_checks.npz"), allow_pickle=False) as data:
        support = data["row_support"]
        matrix = np.unpackbits(data["check_matrix_packed"], axis=1)
    columns = [int.from_bytes(v.tobytes(), "little") for v in np.packbits(matrix.T, axis=1, bitorder="little")]

    branches = {}
    for name, stop in (('guarded', row["pn_interval"][1]), ('to_systematic_end', SYSTEMATIC)):
        first = row["pn_interval"][0]
        padding = (np.arange(LENGTH) >= first) & (np.arange(LENGTH) < stop)
        known = observed | padding
        projected = np.where(observed, prior[index], 0.)
        projected[padding] = np.where(pn[padding], -50., 50.)
        measured, rank, consistent = solve_erasures(columns, np.where(padding, pn, hard[index]), known)
        checkpoints = {n: decode(projected[None], support, "min_sum", n)[0] < 0 for n in (36, 72, 144)}
        branches[name] = (padding, known, measured, rank, consistent, checkpoints)

    with np.load(artifact("v126-mcs66-full-validation", "E128", "mcs66_full_decoder_words.npz"), allow_pickle=False) as data:
        if data["mcs85_frames"][local] != row["frame"] or not data["mcs85_ordinary_pass"][local]:
            raise ValueError("The wideband scoring word is not an accepted same-frame match")
        reference = np.unpackbits(data["mcs85_ordinary_bits"][local]).astype(bool)
    output = {}
    for name, (padding, known, measured, rank, consistent, checkpoints) in branches.items():
        ideal, ideal_rank, ideal_consistent = solve_erasures(columns, np.where(padding, pn, reference), known)
        errors = lambda bits: int(np.count_nonzero(bits != reference)) if bits is not None else None
        output[name] = {"remaining_dimension": int((~known).sum()) - rank,
                        "measured_hard_consistent": consistent, "measured_hard_differences": errors(measured),
                        "measured_visible_errors_outside_padding": int(((reference != hard[index]) & observed & ~padding).sum()),
                        "min_sum": {str(n): {"residual_checks": int(syndrome_count(bits[None], support)[0]),
                                            "bit_differences": errors(bits)} for n, bits in checkpoints.items()},
                        "pn_reference_disagreements": int(((reference != pn) & padding).sum()),
                        "noiseless_compatible": ideal_consistent, "noiseless_erasure_rank": ideal_rank,
                        "noiseless_reference_differences": errors(ideal)}
    result = {"schema": "starlink-leo-padded-code-replay/v1", "selected_geometry": row,
              "mcs85_word_index": index, "visible_bits": int(observed.sum()), "branches": output,
              "input_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    folder = experiment_output("E130")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leo_padded_code_replay.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    for name, branch in output.items():
        print(name, branch)


if __name__ == "__main__":
    main()
