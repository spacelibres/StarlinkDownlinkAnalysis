"""Validate the order of payload allocations against measured constellation changes."""

from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, legacy_artifact, require_run_context
from allocation_core import UNITS
from audit_h2_projection import SELECTION_ROOT, TRACK, WAV_MODEL, band_masks
from audit_ut_binary_tail import BPSK, CROSS32, QAM16, SOURCE, cost, fit_carrier_gain, load_samples, sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_ut_weak_frames import PREFIX
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, equalize_grid, fft_grid, reference_gain
from code9_core import edge_slope_lifts
from layout_core import NONPILOT
from ut_receiver_core import capture_grid
from validate_code37 import constellation_for


LEGACY_RAW = "data/raw/legacy_wideband/Starlink_Downlink_IQ.npy"
ALPHABETS = {
    "BPSK": BPSK, "QPSK": np.exp(1j*np.pi*(.25+.5*np.arange(4))),
    "8QAM": constellation_for(2, False, True), "16QAM": QAM16, "cross32": CROSS32,
}
KINDS = {c: name for name, codes in {
    "BPSK": (3, 5, 9), "QPSK": (10, 12, 21), "8QAM": (27, 28, 37),
    "16QAM": (39, 57, 66, 68, 70), "cross32": (71, 78, 85, 88, 92),
}.items() for c in codes}
WIDTH = {**UNITS, 57: 2048, 66: 4096, 70: 4096, **{k: 3277 for k in (71, 78, 85, 88, 92)}}
ALPHABET_NAMES = tuple(ALPHABETS)
SIZE = 128
SCALE = .03
MIN_RUN = 3
MAX_BRIDGE = 3
RADIUS = 512
MIN_SIDE = 96
MIN_CONTRAST = .005


def measured_costs(flat):
    # Chunk the likelihood calculation without changing a per-resource cost.
    output = np.empty((len(flat), len(ALPHABETS)), dtype=np.float32)
    for first in range(0, len(flat), 8192):
        cut = flat[first:first+8192]
        output[first:first+len(cut)] = np.column_stack([cost(cut, x, SCALE) for x in ALPHABETS.values()])
    return output


def location(costs, center, left, right, radius):
    a, b = max(0, center-radius), min(len(costs), center+radius)
    delta = costs[a:b, ALPHABET_NAMES.index(right)]-costs[a:b, ALPHABET_NAMES.index(left)]
    if len(delta) < 2*MIN_SIDE:
        return None
    sweep = np.r_[0., np.cumsum(delta, dtype=float)]
    split = MIN_SIDE+int(sweep[MIN_SIDE:len(delta)-MIN_SIDE+1].argmax())
    before = float(delta[split-MIN_SIDE:split].mean())
    after = float(delta[split:split+MIN_SIDE].mean())
    return {"coordinate": a+split, "signed_cost_before": before, "signed_cost_after": after,
            "passed_contrast": before > MIN_CONTRAST and after < -MIN_CONTRAST}


def blind_changes(costs):
    n = len(costs)//SIZE
    coarse = costs[:n*SIZE].reshape(n, SIZE, -1).mean(axis=1)
    labels = uniform_filter1d(coarse, size=3, axis=0, mode="nearest").argmin(axis=1)
    edges = np.r_[0, np.flatnonzero(np.diff(labels))+1, len(labels)]
    changes = []
    runs = [(lo, hi, int(labels[lo])) for lo, hi in zip(edges, edges[1:]) if hi-lo >= MIN_RUN]
    # A short window straddling a real change can prefer a third alphabet.
    # Locate the change from the two sustained runs on either side.
    for (_, end, before), (start, _, after) in zip(runs, runs[1:]):
        if (start-end) > MAX_BRIDGE or before == after:
            continue
        center = int(start+end)*SIZE//2
        left, right = ALPHABET_NAMES[before], ALPHABET_NAMES[after]
        found = location(costs, center, left, right, 3*SIZE)
        if found is not None and found["passed_contrast"]:
            changes.append({"coarse": center, "skipped_bridge_blocks": int(start-end),
                            "from": left, "to": right, **found})
    return changes, labels


def allocations(start, entries):
    rows = []
    for c in entries:
        code, n = c["mcs_candidate"], c["quantity"]
        stop = start+n*WIDTH[code]
        rows.append((start, stop, KINDS[code]))
        start = stop
    return rows


def collapsed(rows):
    combined = []
    for lo, hi, name in rows:
        if combined and combined[-1][2] == name:
            combined[-1] = (combined[-1][0], hi, name)
        else:
            combined.append((lo, hi, name))
    return combined


def cost_sum(cum, rows):
    result = 0.
    for lo, hi, name in rows:
        index = ALPHABET_NAMES.index(name)
        result += cum[hi, index]-cum[lo, index]
    return float(result)


def ordering_scores(costs, start, entries, masks):
    expected = collapsed(allocations(start, entries))
    alternatives, equivalent = [], []
    for i, j in combinations(range(len(entries)), 2):
        clone = entries.copy()
        clone[i], clone[j] = clone[j], clone[i]
        modified = collapsed(allocations(start, clone))
        assert modified[-1][1] == expected[-1][1]
        if modified != expected:
            alternatives.append(((i, j), modified))
        else:
            equivalent.append([i, j])
    out = {}
    for model, carriers in masks.items():
        visible = np.ones(len(costs), dtype=bool) if carriers is None else np.isin(
            NONPILOT[np.arange(len(costs)) % len(NONPILOT)], carriers)
        support = int(visible[start:expected[-1][1]].sum())
        cum = np.vstack([np.zeros(len(ALPHABETS)), np.cumsum(costs*visible[:, None], axis=0, dtype=float)])
        baseline = cost_sum(cum, expected)
        scores = [{"swap": list(ij), "excess_cost_per_visible_resource": (cost_sum(cum, alt)-baseline)/max(support, 1)}
                  for ij, alt in alternatives]
        out[model] = {"visible_resources": support, "same_alphabet_schedule": equivalent, "changed_schedule": scores}
    return out


def evaluate(flat, capture, index, h2, h2_start, masks):
    # This call receives no header positions, classes, field labels or candidate endpoints.
    likelihoods = measured_costs(flat)
    candidates, labels = blind_changes(likelihoods)
    first = h2_start+114*h2["blocks"]
    entries = h2["layout20"]["entries"]
    body = allocations(first, entries)
    if body[-1][1] >= len(flat):
        raise ValueError("Predicted allocation outside a complete frame")
    tail = "BPSK" if capture == "ut250" else "QPSK"
    rows = [(-1, first, "BPSK"), *body, (body[-1][1], len(flat), tail)]
    transitions = []
    for (i, j, left), (_, end, right) in zip(rows, rows[1:]):
        base = {"predicted": j, "from": left, "to": right,
                "boundary_kind": "header_to_body" if j == first else "body_to_tail" if j == body[-1][1] else "internal"}
        if left == right:
            transitions.append({**base, "status": "same_alphabet"})
            continue
        matches = [p for p in candidates if p["from"] == left and p["to"] == right and abs(p["coordinate"]-j) <= 2048]
        local = location(likelihoods, j, left, right, min(RADIUS, j-max(i, 0), end-j))
        nearest = min(matches, key=lambda p: abs(p["coordinate"]-j)) if matches else None
        transitions.append({**base, "status": "evaluated", "independent_blind_match": nearest, "conditioned_local": local})
    return {"capture": capture, "frame": index, "h2_start": h2_start, "descriptor_entries": entries,
            "transition_checks": transitions, "unconditioned_blind_candidates": candidates,
            "blind_labels_by_resource_block": labels.tolist(), "order_controls": ordering_scores(likelihoods, first, entries, masks)}


def receive_ut(cases, waveform, truth, sss, masks):
    for frame in cases["candidates"]:
        if not frame["complete_frame"] or frame["h2"]["crc_ok"] is not True:
            continue
        number = frame["candidate_index"]
        raw, slots = capture_grid(waveform, frame["pss"]["epoch"], frame["sampling_scale"], frame["cfo_hz"])
        symbols = equalize_grid(raw[1:], raw[0], sss, slots[1:])["nonpilot"]
        symbols /= reference_gain(symbols.ravel()[:len(PREFIX)], PREFIX)
        good = [i for i, w in enumerate(cases["words"]) if w["candidate_index"] == number
                and w["initial_parity_disagreements"] == 0]
        gains = fit_carrier_gain(symbols, [cases["words"][i] for i in good], truth[good])
        yield evaluate((symbols/gains[None, :]).ravel(), "ut250", number, frame["h2"], frame["predicted_h2_start"], masks)


def receive_legacy(cases, waveform, sss, masks):
    pilots = PILOTS[TRAIN_PILOTS]
    for frame in cases["frames"]:
        number = frame["legacy_frame_index"]
        h2 = frame["second_header"]
        if h2["crc_ok"] is not True:
            continue
        raw = fft_grid(waveform, number)
        grid = equalize_grid(raw[1:], raw[0], sss, np.arange(2, 302))
        phase, _ = edge_slope_lifts(grid["calibration"][:, 1], NONPILOT, pilots[pilots<0].mean(), pilots[pilots>0].mean())
        flat = (grid["nonpilot"]*np.exp(-1j*phase)).ravel()
        flat /= reference_gain(flat[:len(PREFIX)], PREFIX)
        yield evaluate(flat, "legacy480", number, h2, h2["start_resource"], masks)


def main():
    require_run_context()
    used = {}
    def artifact(run, experiment, name):
        return checked_artifact(run, experiment, name, used)
    u = read(artifact("v022-ut-replay", "E014", "ut_transfer.json"))
    leg = read(artifact("v008-complete-header-layout-02", "E010", "full_header_layout.json"))
    ut = ROOT/SOURCE
    legacy = ROOT/LEGACY_RAW
    sss = legacy_artifact("reference/qin/sssVec.mat")
    for path in (ut, legacy, sss, ROOT/TRACK, ROOT/WAV_MODEL,
                 *[SELECTION_ROOT/n/"selection.json" for n in ("DSX01-S05", "paired_RX")]):
        used[path] = sha256(path)
    old_proof = Path("provenance/legacy_research_import.json")
    expected_legacy = next(p["sha256"] for p in read(ROOT/old_proof)["files"] if p["snapshot"] == LEGACY_RAW)
    used[ROOT/old_proof] = sha256(ROOT/old_proof)
    if used[legacy] != expected_legacy or used[sss] != u["sss_reference_sha256"]:
        raise ValueError("Changed wideband IQ or SSS reference")
    s = loadmat(sss)["sssVecFull"].ravel()[LOADED % 1024]
    ideal, _ = band_masks()
    masks = {"full_grid": None, **{k:v for k,v in ideal.items() if "union" not in k}}
    with np.load(artifact("v022-ut-replay", "E014", "ut_observations.npz"), allow_pickle=False) as decoded:
        rows = list(receive_ut(u, load_samples(ut, u), decoded["decoded_code5_words"], s, masks))
    rows.extend(receive_legacy(leg, np.load(legacy, mmap_mode="r", allow_pickle=False), s, masks))
    result = {"schema": "starlink-internal-allocation-audit/v1",
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p,h in used.items()},
              "normalized_cost_scale": SCALE, "blind_block_resources": SIZE, "blind_min_run_blocks": MIN_RUN,
              "blind_max_bridge_blocks": MAX_BRIDGE, "minimum_change_contrast": MIN_CONTRAST, "frames": rows,
              "scope": "Same measured captures. Constellation-only changes and conditional projection tests, not full field semantics."}
    output = experiment_output("E050")
    output.mkdir(parents=True, exist_ok=True)
    (output/"internal_allocations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Tested", len(rows), "complete measured frames", flush=True)


if __name__ == "__main__":
    main()
