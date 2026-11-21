"""Re-encode fully checked MCS 66/85 words and test withheld real symbols."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_mcs_checks import BIT_LENGTH
from audit_cross32_partial_decode import decode, raw_inputs, raw_record_metrics
from audit_mcs66_partial_decoder import GUARD_BITS, RAW_VARIANCE, constellation_llr, syndrome_count
from audit_qam_pn15 import collect, equality_predictions
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_full_code import SYSTEMATIC


CHECKPOINTS = (6, 18, 36)
MASK_SEED = 20261121129
MASK_DIVISOR = 16


def invert_binary(matrix):
    """Return rows of a square GF(2) inverse as little-endian integers."""
    size = len(matrix)
    if matrix.shape != (size, size):
        raise ValueError("Expected square parity matrix")
    rows = [int.from_bytes(b.tobytes(), "little") | (1 << (size + i))
            for i, b in enumerate(np.packbits(matrix, axis=1, bitorder="little"))]
    for j in range(size):
        bit = 1 << j
        p = next((i for i in range(j, size) if rows[i] & bit), None)
        if p is None:
            raise ValueError("Parity matrix is singular")
        rows[j], rows[p] = rows[p], rows[j]
        for i in range(j + 1, size):
            if rows[i] & bit:
                rows[i] ^= rows[j]
    for j in range(size - 1, -1, -1):
        bit = 1 << j
        for i in range(j):
            if rows[i] & bit:
                rows[i] ^= rows[j]
    return [row >> size for row in rows]


def encode(systematic, support, inv):
    rhs = np.stack([np.bitwise_xor.reduce(systematic[:, row[row < SYSTEMATIC]], axis=1) for row in support],
                   axis=1)
    values = [int.from_bytes(r.tobytes(), "little") for r in np.packbits(rhs, axis=1, bitorder="little")]
    parity = np.array([[(row & v).bit_count() & 1 for row in inv] for v in values], dtype=bool)
    return np.column_stack((systematic, parity))


def decode_words(prior, support):
    first_pass = np.zeros(len(prior), dtype=int)
    decisions = prior < 0
    for it in CHECKPOINTS:
        active = np.flatnonzero(first_pass == 0)
        if not len(active):
            break
        words = decode(prior[active], support, "min_sum", it) < 0
        decisions[active] = words
        first_pass[active[syndrome_count(words, support) == 0]] = it
    return decisions, first_pass


def remodulate(bits, points, labels):
    width = labels.shape[1]
    weights = 1 << np.arange(width - 1, -1, -1)
    table = points[np.argsort(labels @ weights)]
    complete = (bits.shape[1] // width) * width
    return table[bits[:, :complete].reshape(len(bits), -1, width) @ weights]


def replay(samples, points, labels, prior, hard, known, pn, frames, support, inv, withheld):
    width = labels.shape[1]
    from_signal = prior.copy()
    if len(withheld):
        indices = (withheld[:, None] * width + np.arange(width)).ravel()
        from_signal[:, indices] = 0.
    decoded, iterations = decode_words(from_signal, support)
    passed = iterations > 0
    if not passed.any():
        raise ValueError("No measured word passed the full matrix")
    regenerated = encode(decoded[passed, :SYSTEMATIC], support, inv)
    if not np.array_equal(regenerated, decoded[passed]):
        raise ValueError("Independent systematic encoder does not agree with the decoder")
    used = samples[passed, :BIT_LENGTH // width]
    predicted = remodulate(regenerated, points, labels)
    changed = regenerated.copy()
    changed[:, SYSTEMATIC:] = np.roll(changed[:, SYSTEMATIC:].reshape(len(changed), -1, 512), 1, axis=-1).reshape(
        len(changed), BIT_LENGTH - SYSTEMATIC)
    wrong = remodulate(changed, points, labels)
    score = withheld if len(withheld) else np.arange((SYSTEMATIC + width - 1) // width, BIT_LENGTH // width)
    prediction = predicted[:, score]
    observed = used[:, score]
    controls = wrong[:, score]
    hard_idx = nearest_symbols(observed, points)
    predicted_idx = nearest_symbols(prediction, points)
    power = (abs(observed) ** 2).sum(axis=1)
    match = hard_idx == predicted_idx
    reports = []
    for i, frame in enumerate(frames):
        row = {"frame": int(frame), "first_full_syndrome_pass_iteration": int(iterations[i]),
               "residual_checks": int(syndrome_count(decoded[i:i + 1], support)[0]),
               "known_pn_bits": int(known[i].sum()), "hard_pn_disagreements": int(((hard[i] != pn) & known[i]).sum()),
               "decoded_pn_disagreements": int(((decoded[i] != pn) & known[i]).sum())}
        reports.append(row)
    for i, local in enumerate(np.flatnonzero(passed)):
        reports[local].update(waveform_symbols=len(score), hard_symbol_matches=int(match[i].sum()),
                              reencoded_symbol_mse=float((abs(prediction[i] - observed[i]) ** 2).sum() / power[i]),
                              shifted_parity_mse=float((abs(controls[i] - observed[i]) ** 2).sum() / power[i]))
    return decoded, {"rows": reports, "passed": int(passed.sum()), "words": len(prior),
                     "withheld_modulation_symbols": withheld.tolist(),
                     "roundtrip_encoder_matches": len(regenerated)}


def inputs66(art, reserved):
    name = "v026-ut-qam-extract-repeat"
    received = art(name, "E028", "ut_qam_symbols.npz")
    meta = read(art(name, "E028", "ut_qam_extraction.json"))
    records, points, mapping = collect(received, meta)
    if meta["reserved_frames"] != reserved:
        raise ValueError("Raw 16QAM and 32QAM split differs")
    bits, _, y, rows = records[66]
    frames = np.array([r["frame"] for r in rows])
    known = np.zeros_like(bits, dtype=bool)
    witnesses = read(art("v086-mcs66-decoder", "E088", "mcs66_partial_decoder.json"))["raw_iq"]["word_observations"]
    for w in witnesses:
        i = w["candidate_index"]
        if w["frame"] != frames[i]:
            raise ValueError("Frozen PN word coordinates disagree")
        known[i, w["bound"] + GUARD_BITS:SYSTEMATIC] = True
    likelihood = constellation_llr(y, points, mapping, RAW_VARIANCE).reshape(bits.shape).astype(np.float32)
    return y, points, mapping, frames, likelihood, bits, known


def inputs85(art, labels):
    path = art("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    points = np.array([complex(r["i"], r["q"]) for r in labels["cross32_mapping"]]) / np.sqrt(20)
    mapping = labels_as_bits(np.array([r["label"] for r in labels["cross32_mapping"]]), 5)
    with np.load(path, allow_pickle=False) as z:
        frames, prior, bits, known = raw_inputs(85, z, labels, SYSTEMATIC)
        y = z["mcs85_symbols"]
    return y, points, mapping, frames, prior, bits, known


def main():
    require_run_context()
    inputs, packed = {}, {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model = art("v126-mcs66-full-code", "E128", "mcs66_full_checks.npz")
    with np.load(model, allow_pickle=False) as z:
        support = z["row_support"]
        parity = np.unpackbits(z["check_matrix_packed"][:, SYSTEMATIC // 8:], axis=1)
    inv = invert_binary(parity)
    labels = read(art("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    reserved = labels["frame_split"]["reserved"]
    rng = np.random.default_rng(MASK_SEED)
    results = {}
    for code, data in ((66, inputs66(art, reserved)), (85, inputs85(art, labels))):
        samples, points, mapping, frames, prior, bits, known = data
        hold = np.isin(frames, reserved)
        eligible = np.arange((SYSTEMATIC + mapping.shape[1] - 1) // mapping.shape[1], BIT_LENGTH // mapping.shape[1])
        positions = np.sort(rng.choice(eligible, len(eligible) // MASK_DIVISOR, replace=False))
        info = {}
        for name, hidden in (("ordinary", np.array([], dtype=int)), ("masked", positions)):
            values, report = replay(samples[hold], points, mapping, prior[hold], bits[hold], known[hold], pn,
                                    frames[hold], support, inv, hidden)
            packed[f"mcs{code}_{name}_bits"] = np.packbits(values, axis=1)
            packed[f"mcs{code}_{name}_pass"] = np.array([row["first_full_syndrome_pass_iteration"] > 0
                                                       for row in report["rows"]])
            info[name] = report
            print("MCS", code, name, report["passed"], "/", report["words"], "zero-syndrome words", flush=True)
            if code == 85 and name == "ordinary" and report["passed"] == report["words"]:
                decoded = bits.copy()
                decoded[hold] = values
                r = read(art("v050-mcs85-spacing", "E052", "mcs85_records.json"))
                info["raw_record_scores"] = raw_record_metrics(r, frames, bits, {"full": decoded}, pn, SYSTEMATIC)
        packed[f"mcs{code}_frames"] = frames[hold]
        results[code] = info
    output = {"schema": "starlink-mcs66-full-code-validation/v1", "checkpoints": CHECKPOINTS,
              "raw_split_seed": 2026112128, "reserved_frames": reserved, "withheld_seed": MASK_SEED,
              "withheld_divisor": MASK_DIVISOR, "results": results,
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    path = experiment_output("E128")
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "mcs66_full_decoder_words.npz", **packed)
    (path / "mcs66_full_decoder_validation.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n",
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
