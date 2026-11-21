"""Transfer a leading Code 5 edge estimate and audit further MCS 6 byte structure."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_qin_mcs6_structure import GUARD, K, QUARTERS, record_candidate
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_qin_low_mcs import target_symbols
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ArchiveReader, ETAG, SIZE, URL
from sparse_code5_core import normalized_min_sum


SEED = 2026112138
ANCHOR = bytes.fromhex("bc6b50ed00d6a2cce0cdff0b")
ANCHOR_OFFSETS = range(17, 21)
EDGE_CARRIER = 511
H1_WIDTH = 228
CODE5_WIDTH = 2048
SLOTS = np.arange(2, 302)


def edge_indices(start, length):
    bins = (start + np.arange(length)) % len(NONPILOT)
    return np.flatnonzero(NONPILOT[bins] == EDGE_CARRIER)


def unique_completion(hard, edge, h):
    erased = hard.copy()
    erased[edge] = 0
    residual = (h @ erased) & 1
    enumerated = ((np.arange(2**len(edge))[:, None] >> np.arange(len(edge))) & 1).astype(np.uint8)
    valid = enumerated[np.all(((enumerated @ h[:, edge].T) & 1) == residual, axis=1)]
    if len(valid) != 1:
        return None, len(valid)
    erased[edge] = valid[0]
    return erased, 1


def code5_references(symbols, h, p):
    """Hide the entire edge carrier from both Code 5 decoders."""
    reference = {"exact_erasure": [], "min_sum": []}
    rows = []
    for i, word in enumerate(symbols):
        edge = edge_indices(H1_WIDTH + i*CODE5_WIDTH, CODE5_WIDTH)
        hard = (word.real < 0).astype(np.uint8)
        exact, solutions = unique_completion(hard, edge, h)
        llr = word.real.copy()
        llr[edge] = 0.
        soft = normalized_min_sum(llr, h)
        rows.append({"word": i, "edge_positions": edge.tolist(), "exact_solutions": solutions,
                     "min_sum_converged": bool(soft["converged"])})
        for branch, complete in (("exact_erasure", exact), ("min_sum", soft["bits"] if soft["converged"] else None)):
            if complete is None:
                continue
            if not np.array_equal((complete[:1024] @ p) & 1, complete[1024:]):
                raise ValueError("Code 5 completion failed independent re-encoding")
            signed = word[edge] * (1 - 2*complete[edge].astype(float))
            reference[branch].extend(signed.tolist())
            difference = hard != complete
            difference[edge] = False
            rows[-1][branch] = {"edge_bits": complete[edge].tolist(),
                                "changed_outside_edge": np.flatnonzero(difference).tolist()}
    return {key: phase_from_reference(np.array(values, dtype=complex)) for key, values in reference.items()}, rows


def phase_from_reference(reference):
    if len(reference) < 2:
        return {"status": "insufficient", "fit_symbols": len(reference)}
    scores = (reference.sum() * QUARTERS).real
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-6, atol=1e-6))
    fit = {"status": "tied" if len(best) != 1 else "estimated",
           "fit_symbols": len(reference), "scores": scores.tolist(),
           "threshold": float(0.5*np.median(abs(reference)))}
    if len(best) == 1:
        fit["turn"] = int(best[0])
    return fit


def transfer_word(symbols, start, fit, pad, expected_pn):
    edge = edge_indices(start, len(symbols))
    observed = symbols[edge]
    labels = np.full(len(edge), -1, dtype=int)
    if pad["accepted"]:
        inside = (edge >= pad["start"] + GUARD) & (edge < min(K, pad["stop"]) - GUARD)
        labels[inside] = expected_pn[edge[inside]]
    result = {"coordinates": edge.tolist(), "known_reference": labels.tolist(),
              "raw_hard": (observed.real < 0).astype(int).tolist()}
    for branch, r in fit.items():
        if r["status"] != "estimated":
            result[branch] = {"status": r["status"]}
            continue
        answers = {}
        for name, turn in (("fixed_zero_turn", 0), ("estimated", r["turn"]), ("wrong_half_turn", (r["turn"] + 2) % 4)):
            receptor = (observed * QUARTERS[turn]).real
            bits = np.where(abs(receptor) <= r["threshold"], -1, receptor < 0)
            answers[name] = bits.tolist()
        result[branch] = {"turn": r["turn"], **answers}
    return result


def positions_of(source, needle):
    starts = []
    start = source.find(needle)
    while start >= 0:
        starts.append(start)
        start = source.find(needle, start+1)
    return starts


def check_anchor(source, stop):
    hits = positions_of(source, ANCHOR)
    return {"all_exact_starts": hits,
            "first_exact_at_frozen_offset": bool(hits and hits[0] - stop in ANCHOR_OFFSETS),
            "within_frozen_window": [x for x in hits if x-stop in ANCHOR_OFFSETS],
            "preceding_low_three": [source[x-1] & 7 if x else None for x in hits]}


def check_records(source, grammar, rng, other_phases):
    records = grammar["records"]
    e_records = []
    for i, row in enumerate(records):
        if row["tag"] == 0xE:
            follow = records[i+1]["tag"] if i+1 < len(records) else grammar["stop_tag"]
            e_records.append({"start": row["start"], "low_bit": source[row["start"]+1] & 1, "following_tag": follow})
    stop = grammar["stop_byte"]
    permuted = rng.permutation(np.frombuffer(source, dtype=np.uint8)).tobytes()
    return {"grammar_matched": grammar["match"], "grammar_stop": stop, "e_records": e_records,
            "anchor": check_anchor(source, stop),
            "control_permuted_bytes": check_anchor(permuted, stop),
            "control_wrong_pn": {key: check_anchor(value, stop) for key, value in other_phases.items()}}


def main():
    require_run_context()
    inputs = {}
    def artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, inputs)
    survey = read(artifact("v034-qin-header-survey", "E036", "qin_header_survey.json"))
    extract = read(artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_extract.json"))
    observed = artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_symbols.npz")
    edges = read(artifact("v034-qin-low-mcs-audit", "E036", "qin_low_mcs_audit.json"))
    prior = read(artifact("v035-qin-mcs6-structure", "E037", "qin_mcs6_structure.json"))
    with np.load(artifact("v010-code5-parity-02", "E012", "code5_observations.npz"), allow_pickle=False) as data:
        p = data["parity_generator"]
    with np.load(artifact("v011-sparse-code5", "E013", "sparse_code5_arrays.npz"), allow_pickle=False) as data:
        h = data["check_matrix"]
    if prior["carrier_audit"]["carrier"] != EDGE_CARRIER:
        raise ValueError("E037 selected carrier changed")
    rows = edges["formats"]["6"]["words"]
    frame_ids = sorted({row["frame"] for row in rows})
    p_slot = np.column_stack([pn_reference(int(k), SLOTS) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    reader = ArchiveReader(ROOT / ".cache" / "qin-mat-pages", False)
    source, source_index = [], []
    estimates = {}
    with reader, h5py.File(reader, "r") as archive:
        for frame, group, flat in target_symbols(archive["yDataDec"], frame_ids, p_slot, signs):
            stop = survey["frames"][frame]["h2_start"]
            leading = flat[H1_WIDTH:stop].reshape(-1, CODE5_WIDTH)
            fit, code5 = code5_references(leading, h, p)
            estimates[frame] = {"group": group, "phases": fit, "code5": code5}
            source.extend(leading)
            source_index.extend([frame] * len(leading))
    old_pages = {row["bytes"][0]: row["sha256"] for row in extract["archive"]["pages"]}
    for start, receipt in reader.receipts.items():
        if old_pages.get(start) != receipt["sha256"]:
            raise ValueError(f"Archive range at {start} differs from E036")
    rng = np.random.default_rng(SEED)
    sequence = pn_bits()
    index = np.arange(K)
    model = edges["pn_model"]
    pn = equality_predictions(index, sequence, model)
    controls = {f"{shift:+d}": equality_predictions(index, sequence, {**model, "phase": model["phase"] + shift})
                for shift in (-1, 1)}
    words = []
    with np.load(observed, allow_pickle=False) as data:
        for i, (row, word, start, frame, previous) in enumerate(zip(
                rows, data["mcs6_symbols"], data["mcs6_resource_start"], data["mcs6_frames"], prior["words"], strict=True)):
            if row["frame"] != frame or previous["word"] != i:
                raise ValueError("E036 and E037 word references no longer correspond")
            hard = np.asarray(word[:K].real < 0, dtype=np.uint8)
            source_word = np.packbits(hard ^ pn).tobytes()
            grammar = record_candidate(source_word)
            if grammar != previous["tests"]["fixed"]:
                raise ValueError("E037 fixed parser changed")
            wrong = {key: np.packbits(hard ^ value).tobytes() for key, value in controls.items()}
            words.append({"word": i, "frame": int(frame), "calibration": row["calibration"],
                          "records": check_records(source_word, grammar, rng, wrong),
                          "carrier_transfer": transfer_word(word, start, estimates[int(frame)]["phases"], row["fixed"], pn)})
    output = experiment_output("E038")
    output.mkdir(parents=True, exist_ok=True)
    result = {"schema": "starlink-qin-mcs6-transfer/v1", "group_seed": edges["whole_group_seed"],
              "groups": edges["formats"]["6"]["groups"], "permutation_seed": SEED,
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "archive": {"url": URL, "etag": ETAG, "bytes": SIZE,
                          "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
              "record_anchor_hex": ANCHOR.hex(), "anchor_start_offsets": list(ANCHOR_OFFSETS),
              "edge_carrier": EDGE_CARRIER, "frames": estimates, "words": words}
    (output / "qin_mcs6_transfer.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(output / "qin_code5_source.npz", frames=source_index, normalized_words=source)
    print(len(estimates), "frames", len(source), "leading Code 5 words", len(words), "MCS 6 candidates", flush=True)


if __name__ == "__main__":
    main()
