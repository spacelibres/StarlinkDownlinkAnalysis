"""Estimate offset-grid synchronism and audit its common first header."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, legacy_artifact, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_ut_weak_frames import CP, FFT_SIZE, PILOT, PREFIX, RATE, SLOT, SMOOTH_BINS, norm_match
from code5_core import LOADED, PILOTS, TRAIN_PILOTS
from receiver_models import pn_reference
from ut_receiver_core import capture_grid, decode_variable_pdu


SEED = 2026112144
SLOTS = np.arange(2, 221)
FFT_TEMPORAL = 8192
HOLD_PILOTS = np.setdiff1d(np.arange(len(PILOTS)), TRAIN_PILOTS)
CAL_SLOTS = (SLOTS >= 9) & (SLOTS <= 110)
TEST_SLOTS = (SLOTS >= 121) & (SLOTS <= 218)
HEADER_LENGTH = 228
NONPREFIX = slice(len(PREFIX), HEADER_LENGTH)
PN = np.column_stack([pn_reference(int(k), SLOTS) for k in LOADED])
WRONG_PILOT = np.column_stack([pn_reference(int(k), SLOTS+1) for k in PILOTS])


def read_grid(iq, record, scale, offset=0.):
    epoch, frequency = record["epoch_480_samples"], record["cfo_hz"]
    last = int(np.ceil(epoch + (SLOT*220 + CP + FFT_SIZE)*scale)) + 12
    x, indices = capture_grid(iq[:last], epoch, scale, frequency+offset)
    if not np.array_equal(indices[1:], SLOTS):
        raise ValueError("Unexpected offset-grid slot support")
    return x


def temporal_power(pilot_symbols, slots):
    choose = (SLOTS >= 9) & (SLOTS <= 198)
    values = pilot_symbols[choose][:, slots]
    spectrum = fft.fftshift(fft.fft(values, n=FFT_TEMPORAL, axis=0), axes=0)
    energy = len(values)*np.sum(abs(values)**2, axis=0)
    return abs(spectrum)**2/np.maximum(energy, 1e-30)


def timing_fit(bank, train):
    hz = fft.fftshift(fft.fftfreq(FFT_TEMPORAL, SLOT/RATE))
    selected = abs(hz) <= 20_000
    hz = hz[selected]
    power = np.mean([temporal_power(bank[t], TRAIN_PILOTS)[selected] for t in train], axis=0)
    sign = PILOTS[TRAIN_PILOTS] > 0
    sides = np.stack([power[:, ~sign].mean(axis=1), power[:, sign].mean(axis=1)])
    peaks, height = [], []
    for values in sides:
        i = int(values.argmax())
        location = hz[i]
        if 0 < i < len(values)-1:
            left, center, right = values[i-1:i+2]
            curvature = left-2*center+right
            if curvature < 0:
                location += .5*(left-right)/curvature*(hz[1]-hz[0])
        peaks.append(float(location))
        height.append(float(values[i]))
    centers = [PILOTS[TRAIN_PILOTS][~sign].mean(), PILOTS[TRAIN_PILOTS][sign].mean()]
    slope, intercept = np.polyfit(centers, peaks, 1)
    return {"pilot_center_subcarriers": centers, "pilot_peak_hz": peaks, "pilot_peak_power": height,
            "cfo_residual_hz": float(intercept), "sampling_scale": float(1-slope/(RATE/FFT_SIZE))}


def soft_resources(grid, sss):
    h = grid[0]/sss
    channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(h, 2)])
    body = grid[1:]*PN.conj()
    pilot_channel = np.broadcast_to(channel[PILOT], (len(SLOTS), len(PILOTS)))
    pilot_gain = np.sum(body[CAL_SLOTS][:, PILOT][:, TRAIN_PILOTS] *
                        pilot_channel[CAL_SLOTS][:, TRAIN_PILOTS].conj())
    if abs(pilot_gain) == 0:
        raise ValueError("Zero pilot gain")
    factor = pilot_gain.conj()/abs(pilot_gain)
    observation = body[TEST_SLOTS][:, PILOT][:, HOLD_PILOTS].ravel()
    target = pilot_channel[TEST_SLOTS][:, HOLD_PILOTS].ravel()
    offset_pn = grid[1:, PILOT]*WRONG_PILOT.conj()
    metrics = {
        "known_pn": norm_match(observation, target),
        "shifted_pn": norm_match(offset_pn[TEST_SLOTS][:, HOLD_PILOTS].ravel(), target),
    }
    weighted = (grid[1:, ~PILOT]*channel[~PILOT].conj()*factor).ravel()
    derotated = weighted*PN[:, ~PILOT].conj().ravel()
    length = 10_040
    scale = np.linalg.norm(derotated[:HEADER_LENGTH])
    return metrics, derotated[:length]/scale, weighted[:length]/scale


def compact(header):
    return {key: value for key, value in header.items() if key not in ("bits", "reencoded")}


def make_h2_query(vector, h1):
    if h1["crc_ok"] is not True:
        return None, None
    entry = h1["layout20"]["entries"][0]
    if entry["mcs_candidate"] != 5 or entry["quantity"] < 1:
        return None, None
    start = HEADER_LENGTH+2048*entry["quantity"]
    if start+114 > len(vector):
        return None, None
    return start, compact(decode_variable_pdu(vector[start:start+798], "h2"))


def prefix_rotation(vector):
    gain = vector[:len(PREFIX)] @ PREFIX
    return gain.conj()/abs(gain)


def parity_audit(vectors, start, checks, seed):
    words = vectors[:, HEADER_LENGTH:start].real.reshape(len(vectors), -1, 2048)
    z = np.tanh(.8*words/np.sqrt(np.mean(words**2, axis=-1, keepdims=True)))
    if not np.all(checks.sum(axis=1) == 6):
        raise ValueError("Unexpected sparse support")
    columns = np.nonzero(checks)[1].reshape(len(checks), 6)
    def score(indices):
        return np.prod(z[..., indices], axis=-1).mean(axis=(1, 2))
    rng = np.random.default_rng(seed)
    shifts = rng.choice(np.arange(2, 2047), 199, replace=False)
    held = score(columns)
    null = [float(np.mean(score((columns+shift) % 2048))) for shift in shifts]
    return {"soft_mean_by_frame": held.tolist(), "soft_mean": float(np.mean(held)),
            "neighbor_means": {str(i): float(np.mean(score((columns+i) % 2048))) for i in (-1, 1)},
            "rotation_shifts": shifts.tolist(), "rotation_null_means": null}


def evaluate_headers(raw, no_pn, train, reserved, seed, checks):
    train_soft = np.stack([raw[t] for t in train]).mean(axis=0)
    phase = prefix_rotation(train_soft)
    fit = decode_variable_pdu((phase*train_soft)[:HEADER_LENGTH], "h1")
    test = np.stack([phase*raw[t] for t in reserved])
    hold = test.mean(axis=0)
    verified = decode_variable_pdu(hold[:HEADER_LENGTH], "h1")
    stripped = phase*np.stack([no_pn[t] for t in reserved]).mean(axis=0)
    start, h2 = make_h2_query(hold, fit)
    # A second receive branch uses the reserved group's known prefix as its
    # training field. The other 180 encoded resources cannot set the rotation.
    local_phase = prefix_rotation(hold)
    local = local_phase*hold
    out = {"h1_calibration": compact(fit), "h1_held": compact(verified),
           "no_pn_held_h1": compact(decode_variable_pdu(stripped[:HEADER_LENGTH], "h1")),
           "h2_start": start, "h2_held": h2, "prefix_phase_used_rad": float(np.angle(phase)),
           "known_prefix_added_phase_rad": float(np.angle(local_phase)),
           "h1_held_known_prefix": compact(decode_variable_pdu(local[:HEADER_LENGTH], "h1")),
           "h2_held_known_prefix": make_h2_query(local, fit)[1],
           "no_pn_held_h1_known_prefix": compact(decode_variable_pdu(
               (prefix_rotation(stripped)*stripped)[:HEADER_LENGTH], "h1")),
           "h1_per_frame": {str(t): compact(decode_variable_pdu(z[:HEADER_LENGTH], "h1"))
                            for t, z in zip(reserved, test, strict=True)}}
    if "reencoded" in fit:
        sign = (1-2*fit["reencoded"].astype(float))[NONPREFIX]
        values = hold[NONPREFIX].real
        den = np.linalg.norm(values)*np.linalg.norm(sign)
        rng = np.random.default_rng(seed)
        groups = sign.reshape(-1, 12)
        null = [float(values @ rng.permutation(groups).ravel()/den) for _ in range(399)]
        out["nonprefix_check"] = {
            "held_normalized_signed_match": float(values @ sign/den),
            "held_hard_matches": int(np.sum(values*sign > 0)),
            "held_totals": len(sign),
            "frame_signed_matches": ((test[:, NONPREFIX].real @ sign) /
                                     (np.linalg.norm(test[:, NONPREFIX].real, axis=1)*np.linalg.norm(sign))).tolist(),
            "group_permutation_scores": null,
        }
    if start is not None:
        out["h2_per_frame"] = {str(t): make_h2_query(z, fit)[1] for t, z in zip(reserved, test, strict=True)}
        out["code5_held_soft_checks"] = parity_audit(test, start, checks, seed+1)
    return out, test, phase


def main():
    require_run_context()
    inputs = {}
    def artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, inputs)
    record = read(artifact("v041-ut-dual-grid", "E043", "ut_weak_frames.json"))
    reference = read(artifact("v022-ut-replay", "E014", "ut_transfer.json"))
    parity_file = artifact("v011-sparse-code5", "E013", "sparse_code5_arrays.npz")
    raw = ROOT/SOURCE
    iq = load_samples(raw, reference)
    inputs[raw] = reference["source_sha256"]
    sss_file = legacy_artifact("reference/qin/sssVec.mat")
    inputs[sss_file] = sha256(sss_file)
    if inputs[sss_file] != reference["sss_reference_sha256"]:
        raise ValueError("Unexpected SSS reference hash")
    sss = loadmat(sss_file)["sssVecFull"].ravel()[LOADED % 1024]
    with np.load(parity_file, allow_pickle=False) as saved:
        checks = saved["check_matrix"]
    found_a = {r["tick"] for r in record["records"] if r["sync"]["sync_pair_passed"]}
    b = {r["tick"]: r["sync"] for r in record["offset_grid"]["records"]
         if r.get("sync", {}).get("sync_pair_passed")}
    order = np.random.default_rng(SEED).permutation(sorted(set(b)-found_a)).tolist()
    overlap = sorted(set(b)&found_a)
    pilot_base, prior_scores = {}, {}
    prior = record["offset_grid"]["sampling_scale"]
    for tick, sync in b.items():
        grid = read_grid(iq, sync, 1.)
        pilot_base[tick] = grid[1:, PILOT]*PN[:, PILOT].conj()
        prior_scores[tick] = soft_resources(read_grid(iq, sync, prior), sss)[0]
    results, saved = [], {}
    for index, (train, reserve) in enumerate(((order[:7], order[7:]), (order[7:], order[:7]))):
        fit = timing_fit(pilot_base, train)
        scale, residual = fit["sampling_scale"], fit["cfo_residual_hz"]
        observations, soft, no_pn = {}, {}, {}
        for tick, sync in b.items():
            values = read_grid(iq, sync, scale, residual)
            observations[tick], soft[tick], no_pn[tick] = soft_resources(values, sss)
        limits, held_data, _ = evaluate_headers(soft, no_pn, train, reserve, SEED+10*index, checks)
        overlap_limits, overlap_data, _ = evaluate_headers(soft, no_pn, train, overlap, SEED+10*index+4, checks)
        results.append({"train_ticks": train, "held_ticks": reserve, "timing": fit, "by_tick_pilot": observations,
                        "main_headers": limits, "overlap_headers": overlap_limits})
        saved.update({f"fold_{index}_held": held_data, f"fold_{index}_overlap": overlap_data,
                      f"fold_{index}_train": np.stack([soft[t] for t in train])})
        print("offset-grid fold", index, scale, residual, limits["h1_held"].get("words_hex"),
              limits["h1_held"]["crc_ok"], flush=True)
    output = experiment_output("E044")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "starlink-ut-offset-grid-pilots/v1",
               "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
               "seed": SEED, "reference_slots": [int(SLOTS[0]), int(SLOTS[-1])],
               "pilot_time_spectrum_fft": FFT_TEMPORAL, "offset_grid_prior_scale": prior,
               "prior_by_tick_pilot": prior_scores, "overlap_ticks": overlap,
               "calibration_history": "An exploratory same-capture probe informed the fit and group header hypothesis.",
               "folds": results, "scope": "Group header decoding in a single capture. No single weak frame or valid payload is inferred from pooling."}
    (output/"ut_offset_grid_pilots.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(output/"ut_offset_grid_soft.npz", **saved)


if __name__ == "__main__":
    main()
