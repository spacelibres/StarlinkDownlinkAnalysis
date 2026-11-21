"""Build IQ-backed candidate word arrays for MCS 66 and 70."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_ut_binary_tail import (
    BPSK, CROSS32, QAM16, RUN, SOURCE, cost, fit_carrier_gain, load_samples, sha256,
)
from code28_receiver_core import crossfit_gain
from code5_core import LOADED, equalize_grid, reference_gain
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from ut_receiver_core import capture_grid
from validate_qc_figure import constellation

TAIL = ROOT / "runs/v023-binary-tail-final/artifacts/experiments/E025/results/ut_binary_tail.json"
SSS = ROOT / "references/qin/sssVec.mat"
TARGETS = (66, 70)
WIDTH = 4096
NRESOURCE = len(NONPILOT)
SEED = 2026112128
AXIS64 = np.arange(-7, 8, 2)
QAM64 = (AXIS64[:, None] + 1j * AXIS64[None, :]).ravel() / np.sqrt(42)


def allocation_regions(record, units):
    header = record["h2"]
    start = record["predicted_h2_start"] + header["blocks"] * 114
    regions = []
    for row in header["layout20"]["entries"]:
        code, quantity = row["mcs_candidate"], row["quantity"]
        end = start + units[code] * quantity
        regions.append((code, start, end, quantity))
        start = end
    return regions


def score_alphabets(raw, scale=.03):
    samples = raw[::17]
    return {
        name: float(cost(samples, points, scale).mean())
        for name, points in (("bpsk", BPSK), ("square16", QAM16), ("cross32", CROSS32), ("square64", QAM64))
    }


def record_refinement(flat, code, start, stop, quantity, frame):
    if stop - start != quantity * WIDTH or start < 1 or stop >= len(flat):
        raise ValueError("MCS word grid out of range")
    indices = np.arange(start - 1, stop + 1)
    body = flat[indices]
    frequencies = indices % NRESOURCE
    slots = indices // NRESOURCE
    calibrated = crossfit_gain(body, frequencies, slots, QAM16)
    calibrated = crossfit_gain(calibrated, slots, frequencies, QAM16)
    words = start + np.arange(quantity)[:, None] * WIDTH + np.arange(-1, WIDTH + 1)
    take = words - (start - 1)
    report = {
        "format": code, "frame": frame, "start": start, "end": stop, "quantity": quantity,
        "frozen_front_end_alphabet_costs": score_alphabets(body[1:-1]),
        "mean_nearest_square16_mse": float(np.mean(np.min(abs(body[1:-1, None] - QAM16) ** 2, axis=1))),
        "refined_mean_nearest_square16_mse": float(np.mean(np.min(abs(calibrated[1:-1, None] - QAM16) ** 2, axis=1))),
    }
    return report, body[take], calibrated[take]


def main():
    require_run_context()
    reference_path = RUN / "ut_transfer.json"
    code5_path = RUN / "ut_observations.npz"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    t = json.loads(TAIL.read_text(encoding="utf-8"))["conditional_length_audit"]
    units = {int(k): v for k, v in t["candidate_resource_units"].items()}
    ends = {r["candidate"]: r["template_refined_endpoint"] for r in t["comparisons"]}
    source = ROOT / SOURCE
    samples = load_samples(source, reference)
    sss = loadmat(SSS)["sssVecFull"].ravel()[LOADED % 1024]
    with np.load(code5_path, allow_pickle=False) as file:
        code5_words = file["decoded_code5_words"]
    observations = {code: {"baseline": [], "refined": [], "frames": [], "starts": []} for code in TARGETS}
    rows = []
    p, mapping = constellation(39)
    known_prefix = np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)]
    known_signs = 1 - 2 * encode_block(known_prefix)[:48].astype(float)
    for case in reference["candidates"]:
        frame = case["candidate_index"]
        if frame not in ends:
            continue
        grid, slots = capture_grid(samples, case["pss"]["epoch"], case["sampling_scale"], case["cfo_hz"])
        eq = equalize_grid(grid[1:], grid[0], sss, slots[1:])
        z = eq["nonpilot"] / reference_gain(eq["nonpilot"].ravel()[:48], known_signs)
        used = [i for i, w in enumerate(reference["words"]) if w["candidate_index"] == frame]
        if any(reference["words"][i]["initial_parity_disagreements"] for i in used):
            raise ValueError("Code 5 receiver reference mismatch")
        gain = fit_carrier_gain(z, [reference["words"][i] for i in used], code5_words[used])
        flat = (z / gain[None, :]).ravel()
        regions = allocation_regions(case, units)
        if regions[-1][2] != ends[frame]:
            raise ValueError("Resource sum disagrees with E025")
        for code, start, stop, count in regions:
            if code not in TARGETS:
                continue
            row, initial, refined = record_refinement(flat, code, start, stop, count, frame)
            rows.append(row)
            group = observations[code]
            group["baseline"].extend(initial)
            group["refined"].extend(refined)
            group["frames"].extend([frame] * count)
            group["starts"].extend(start + np.arange(count) * WIDTH)
            print(frame, code, count, row["refined_mean_nearest_square16_mse"], flush=True)
    frames = np.random.default_rng(SEED).permutation(sorted(ends))
    arrays = {"constellation_points": p, "frozen_gray_mapping": mapping}
    for code, record in observations.items():
        for key, value in record.items():
            dtype = np.complex64 if key in ("baseline", "refined") else np.int32
            arrays[f"mcs{code}_{key}"] = np.array(value, dtype=dtype)
    result = {
        "schema": "starlink-ut-qam-words/v1",
        "inputs_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in (source, SSS, reference_path, code5_path, TAIL)},
        "observed_allocations": rows,
        "groups_seed": SEED, "calibration_frames": sorted(frames[:4].tolist()), "reserved_frames": sorted(frames[4:].tolist()),
        "resources_per_candidate_word": WIDTH,
        "scope": "Complete E025 frames, conditional published allocation width, Code 5 aided gain and parity-blind square-QAM crossed correction.",
    }
    out = experiment_output("E028")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ut_qam_extraction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "ut_qam_symbols.npz", **arrays)


if __name__ == "__main__":
    main()
