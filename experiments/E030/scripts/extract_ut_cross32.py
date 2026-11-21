"""Extract other wideband MCS regions using the frozen E025 layout."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_ut_binary_tail import CROSS32, QAM16, RUN, SOURCE, fit_carrier_gain, load_samples, sha256
from code28_receiver_core import crossfit_gain
from code5_core import LOADED, equalize_grid, reference_gain
from educational_fec_demo import encode_block, word_bits
from extract_ut_qam import NRESOURCE, SEED, SSS, TAIL, allocation_regions, score_alphabets
from ut_receiver_core import capture_grid

TARGETS = (57, 71, 78, 85, 88, 92)


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
        code5 = file["decoded_code5_words"]
    known = np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)]
    signs = 1 - 2 * encode_block(known)[:48].astype(float)
    observations = {c: [] for c in TARGETS}
    frame_records = {c: [] for c in TARGETS}
    rows = []
    for case in reference["candidates"]:
        frame = case["candidate_index"]
        if frame not in ends:
            continue
        grid, slots = capture_grid(samples, case["pss"]["epoch"], case["sampling_scale"], case["cfo_hz"])
        z = equalize_grid(grid[1:], grid[0], sss, slots[1:])["nonpilot"]
        z = z / reference_gain(z.ravel()[:48], signs)
        used = [i for i, w in enumerate(reference["words"]) if w["candidate_index"] == frame]
        if any(reference["words"][i]["initial_parity_disagreements"] for i in used):
            raise ValueError("Code 5 receiver reference mismatch")
        gain = fit_carrier_gain(z, [reference["words"][i] for i in used], code5[used])
        flat = (z / gain[None]).ravel()
        regions = allocation_regions(case, units)
        if not regions or regions[-1][2] != ends[frame]:
            raise ValueError("E025 boundary mismatch")
        possible = [r for r in regions if r[0] in TARGETS and r[0] != 57]
        candidates = np.concatenate([np.arange(start, stop) for _, start, stop, _ in possible])
        candidate_slots, candidate_carriers = candidates // NRESOURCE, candidates % NRESOURCE
        corrected = flat.copy()
        values = crossfit_gain(flat[candidates], candidate_carriers, candidate_slots, CROSS32)
        corrected[candidates] = crossfit_gain(values, candidate_slots, candidate_carriers, CROSS32)
        for code, start, stop, count in regions:
            if code not in TARGETS:
                continue
            q = np.arange(start, stop)
            body = flat[q]
            refined = corrected[q]
            observations[code].extend(refined.reshape(count, units[code]))
            frame_records[code].extend([frame] * count)
            rows.append({"format": code, "frame": frame, "start": start, "stop": stop, "count": count,
                         "baseline_costs": score_alphabets(body), "refined_costs": score_alphabets(refined)})
            print(frame, code, count, min(rows[-1]["baseline_costs"], key=rows[-1]["baseline_costs"].get), flush=True)
    arrays = {"cross32_points": CROSS32, "square16_points": QAM16}
    for code in TARGETS:
        arrays[f"mcs{code}_symbols"] = np.asarray(observations[code], dtype=np.complex64)
        arrays[f"mcs{code}_frames"] = np.asarray(frame_records[code], dtype=np.int32)
    split = np.random.default_rng(SEED).permutation(sorted(ends))
    result = {
        "schema": "starlink-ut-cross32-observations/v1",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                          for p in (source, reference_path, code5_path, TAIL, SSS)},
        "split_seed": SEED, "calibration_frames": sorted(split[:4].tolist()), "reserved_frames": sorted(split[4:].tolist()),
        "regions": rows, "scope": "Fixed layout; complete candidates only. Refinement geometrical and no FEC used.",
    }
    out = experiment_output("E030")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ut_cross32_extraction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "ut_cross32_symbols.npz", **arrays)


if __name__ == "__main__":
    main()
