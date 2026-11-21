"""Locate the offset-grid H2 and test deterministic structure in its preceding allocation."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions, ordered_indices
from audit_ut_offset_grid_pilots import HEADER_LENGTH, compact
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import prefix_fields
from receiver_models import pn_bits
from ut_receiver_core import decode_variable_pdu


SEED = 2026112145
SEARCH_HALF_WIDTH = 1000
GROUP_SIZE = 128
REGION_FLOOR = .15
WORD_SIZE = 2048
HALF_WORD = WORD_SIZE//2
TRIALS = 399
BYTE_BANK = np.array([1-2*encode_block(word_bits(byte << 24))[:24].astype(float) for byte in range(256)])


def unit(values):
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return values/np.maximum(length, 1e-30)


def orient_h1(vectors, words):
    h1 = 1-2*np.concatenate([encode_block(word_bits(int(w, 16))) for w in words]).astype(float)
    gains = vectors[:, :HEADER_LENGTH] @ h1
    aligned = vectors * (gains.conj()/np.maximum(abs(gains), 1e-30))[:, None]
    strength = abs(unit(vectors[:, :HEADER_LENGTH]) @ unit(h1))
    return aligned, strength


def byte_survey(pools, start):
    shifts = np.arange(-SEARCH_HALF_WIDTH, SEARCH_HALF_WIDTH+1)
    tables = {}
    for group, vectors in pools.items():
        mean = vectors.real.mean(axis=0)
        window = mean[start+shifts[0]:start+shifts[-1]+24]
        windows = np.lib.stride_tricks.sliding_window_view(window, 24)
        tables[group] = unit(windows) @ unit(BYTE_BANK).T
    output = {}
    center_index = SEARCH_HALF_WIDTH
    for fit, test in (("eight", "seven"), ("seven", "eight"), ("eight", "overlap")):
        table = tables[test]
        center = table[center_index]
        selected = int(np.argmax(tables[fit][center_index]))
        fixed = table[:, selected]
        close = np.r_[fixed[:center_index], fixed[center_index+1:]]
        not_at_center = np.delete(table, center_index, axis=0)
        output[test] = {
            "calibration_group": fit, "selected_byte": f"0x{selected:02X}",
            "held_best_byte": f"0x{int(np.argmax(center)):02X}",
            "held_match": float(center[selected]),
            "best_other_byte_match_at_predicted_start": float(np.max(np.delete(center, selected))),
            "strongest_search_shift_for_fixed_byte": int(shifts[np.argmax(fixed)]),
            "strongest_other_shift_for_fixed_byte": float(np.max(close)),
            "maximum_other_shift_and_any_byte": float(np.max(not_at_center)),
            "held_fixed_byte_match_by_shift": fixed.tolist(),
        }
    return output


def suffix_survey(pools, tick_groups, start):
    # Hypotheses nominated in the exploratory scan. Use separate coded
    # outputs for the frame assignment and the directional replication.
    suffix = np.array([1-2*encode_block(word_bits((0x3400+b)<<16))[24:48].astype(float) for b in range(256)])
    legal = []
    for index in range(256):
        fields = prefix_fields(word_bits((0x3400+index)<<16))
        if 2 <= fields["blocks"] <= 7 and 1 <= fields["entry_count"] and 24+20*fields["entry_count"] <= 32*fields["blocks"]:
            legal.append(index)
    chosen = np.random.default_rng(SEED).permutation(8)[:4]
    fit = np.isin(np.arange(24)//3, chosen)
    difference = suffix[0x44]-suffix[0xCC]
    records = []
    for group, values in pools.items():
        for tick, trace in zip(tick_groups[group], values.real, strict=True):
            region = trace[start+24:start+48]
            prefix_match = float(unit(trace[start:start+24]) @ unit(BYTE_BANK[0x34]))
            first = float(unit(region[fit]) @ unit(difference[fit]))
            second = float(unit(region[~fit]) @ unit(difference[~fit]))
            selected = prefix_match >= .25 and abs(first) >= .25
            order = np.argsort(-(region @ suffix[legal].T), kind="stable")[:3]
            decoded = decode_variable_pdu(trace[start:start+798].astype(complex), "h2")
            records.append({"group": group, "tick": tick, "common_byte_score": prefix_match,
                            "classification_margin": first, "held_margin": second,
                            "direction_assigned": int(np.sign(first)) if selected else 0,
                            "best_legal_suffixes_on_all_symbols": [f"0x{legal[i]:02X}" for i in order],
                            "individual_direct_h2": compact(decoded)})
    return {"fit_trellis_triplets": chosen.tolist(), "held_trellis_triplets": np.unique(np.arange(24)[~fit]//3).tolist(),
            "fixed_pair": ["0x3444", "0x34CC"], "min_common_byte_score": .25, "min_absolute_fit_margin": .25,
            "records": records}


def exhaustive_pn_phases(averages, pn, model):
    """Score every circular LFSR phase with FFT correlation."""
    coords = np.arange(HALF_WORD)
    polarity = -1 if model["complement"] else 1
    reference = fft.rfft(polarity*(1-2*pn.astype(float)))
    records = []
    for name, width, direction in (("nominal", model["within_unit_reverse"], model["direction"]),
                                   ("no_reversal", 1, model["direction"]),
                                   ("wrong_direction", model["within_unit_reverse"], -model["direction"])):
        indices = (direction*ordered_indices(coords, width)) % len(pn)
        banks = {}
        for group, mean in averages.items():
            weights = np.zeros((*mean.shape[:-1], len(pn)))
            weights[..., indices] = mean
            banks[group] = fft.irfft(fft.rfft(weights, axis=-1).conj()*reference, n=len(pn), axis=-1)/HALF_WORD
        for train, test in (("eight", "seven"), ("seven", "eight")):
            for word, side in np.ndindex(banks[train].shape[:2]):
                cal = banks[train][word, side]
                held = banks[test][word, side]
                phase = int(np.argmax(abs(cal)))
                rank = int(np.count_nonzero(abs(held) > abs(held[phase]))+1)
                records.append({"model": name, "train": train, "test": test, "word_index": word, "half": side,
                                "phase": phase, "calibration_score": float(cal[phase]),
                                "held_score_at_calibration_phase": float(held[phase]), "held_rank": rank,
                                "held_absolute_best_phase": int(np.argmax(abs(held)))})
    return records


def pn_windows(pools, model):
    coordinates = np.arange(HALF_WORD)
    pn = pn_bits()
    variants = {"nominal": model, "plus_one_pn_phase": {**model, "phase": model["phase"]+1},
                "minus_one_pn_phase": {**model, "phase": model["phase"]-1},
                "no_bit_reversal": {**model, "within_unit_reverse": 1}}
    sign = {key: 1-2*equality_predictions(coordinates, pn, val).astype(float)
            for key, val in variants.items()}
    rng = np.random.default_rng(SEED+1)
    candidates = np.setdiff1d(np.arange(len(pn)), [m["phase"] for m in variants.values()])
    null_phases = rng.choice(candidates, size=TRIALS, replace=False)
    null = np.stack([1-2*equality_predictions(coordinates, pn, {**model, "phase": n}).astype(float)
                     for n in null_phases])
    averages = {key: values.mean(axis=0) for key, values in pools.items()}
    template = sign["nominal"]
    partition = rng.permutation(HALF_WORD//GROUP_SIZE)
    support = []
    for train, test in (("eight", "seven"), ("seven", "eight")):
        observed = averages[test]
        for word, side in np.ndindex(observed.shape[:2]):
            cal_cells = (averages[train][word, side] * template).reshape(-1, GROUP_SIZE).mean(axis=1)
            prefix = np.r_[0., np.cumsum(cal_cells-REGION_FLOOR)]
            gain = prefix-np.minimum.accumulate(prefix)
            stop = int(gain.argmax())
            first = int(prefix[:stop+1].argmin())
            row = {"train": train, "test": test, "word_index": word, "half": side,
                   "train_cell_signed_means": cal_cells.tolist(), "start": first*GROUP_SIZE, "stop": stop*GROUP_SIZE}
            if stop > first:
                select = slice(first*GROUP_SIZE, stop*GROUP_SIZE)
                t = observed[word, side, select]
                row["held_signed_means"] = {name: float(np.mean(t*s[select])) for name, s in sign.items()}
                row["held_null_means"] = (null[:, select] @ t / len(t)).tolist()
                row["overlap_signed_mean"] = float(np.mean(averages["overlap"][word, side, select] * template[select]))
            support.append(row)

    comparisons = []
    matrix = np.stack([averages[g] for g in ("eight", "seven", "overlap")])
    cells = matrix.reshape(*matrix.shape[:-1], -1, GROUP_SIZE)
    pn_cells = template.reshape(-1, GROUP_SIZE)
    projection = (cells*pn_cells).sum(axis=-1, keepdims=True)/GROUP_SIZE
    residual = (cells-projection*pn_cells).reshape(matrix.shape)
    perms = np.array([rng.permutation(partition) for _ in range(TRIALS)])
    for word, side in np.ndindex(matrix.shape[1:3]):
        raw = unit(matrix[:, word, side])
        remainder = unit(residual[:, word, side])
        shuffled = remainder[0].reshape(-1, GROUP_SIZE)[perms].reshape(TRIALS, HALF_WORD)
        comparisons.append({"word_index": word, "half": side, "cross_group": (raw @ raw.T).tolist(),
                            "after_pn_projection": (remainder @ remainder.T).tolist(),
                            "after_pn_projection_group_permutation_scores": (shuffled @ remainder[1]).tolist()})
    return {"pn_model": model, "half_group_length": GROUP_SIZE, "region_floor": REGION_FLOOR,
            "shifted_pn_control_phases": null_phases.tolist(), "pn_support": support,
            "exhaustive_phase_transfer": exhaustive_pn_phases(averages, pn, model),
            "repeat_comparisons": comparisons}, averages


def main():
    require_run_context()
    inputs = {}
    def checked(run, exp, name):
        return checked_artifact(run, exp, name, inputs)
    e044 = "v042-ut-offset-grid-header"
    report = read(checked(e044, "E044", "ut_offset_grid_pilots.json"))
    source = checked(e044, "E044", "ut_offset_grid_soft.npz")
    known_pn = read(checked("v027-qam-pn15-final", "E029", "qam_pn15_audit.json"))["fit"]["winner"]
    a, b = report["folds"]
    h1, h1_check = [row["main_headers"]["h1_held_known_prefix"] for row in (a, b)]
    if not h1["crc_ok"] or not h1_check["crc_ok"] or h1["words_hex"] != h1_check["words_hex"]:
        raise ValueError("The two prior H1 fits disagree")
    entry = h1["layout20"]["entries"][0]
    if entry["mcs_candidate"] != 5:
        raise ValueError("Expected Code 5")
    count = entry["quantity"]
    start = HEADER_LENGTH+count*WORD_SIZE
    tick_groups = {"eight": a["held_ticks"], "seven": b["held_ticks"], "overlap": report["overlap_ticks"]}
    with np.load(source, allow_pickle=False) as vectors:
        groups = {"eight": vectors["fold_0_held"], "seven": vectors["fold_1_held"], "overlap": vectors["fold_0_overlap"]}
    if set(tick_groups["eight"]) & set(tick_groups["seven"]):
        raise ValueError("Whole held-frame groups overlap")
    aligned, h1_match = {}, {}
    for group, iq in groups.items():
        aligned[group], matches = orient_h1(iq, h1["words_hex"])
        h1_match[group] = matches.tolist()
    regions = {}
    for group, received in aligned.items():
        physical = received[:, HEADER_LENGTH:start].real.reshape(-1, count, 2, HALF_WORD)
        regions[group] = unit(physical)*np.sqrt(HALF_WORD)
    repetition, means = pn_windows(regions, known_pn)
    summary = {"schema": "starlink-ut-offset-h2-systematics/v1", "seed": SEED,
               "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
               "h1_words_hex": h1["words_hex"], "first_allocation": entry, "h2_start_resource": start,
               "tick_groups": tick_groups, "h1_refit_match_by_group": h1_match,
               "h2_start_search_half_width": SEARCH_HALF_WIDTH, "byte_test": byte_survey(aligned, start),
               "suffix_test": suffix_survey(aligned, tick_groups, start), "allocation_test": repetition,
               "scope": "Same 50 ms capture. H2 boundary and partial byte support; no asserted full H2 or Code 5 decode."}
    out = experiment_output("E045")
    out.mkdir(parents=True, exist_ok=True)
    (out/"ut_offset_h2_systematics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(out/"ut_offset_h2_systematics.npz", **{
        f"{group}_h2_soft": values[:, start:start+798] for group, values in aligned.items()
    }, **{f"{key}_code5_half_word_means": values for key, values in means.items()})
    print("H2 byte and timing", [(g, m["selected_byte"], m["strongest_search_shift_for_fixed_byte"])
                                 for g, m in summary["byte_test"].items()])


if __name__ == "__main__":
    main()
