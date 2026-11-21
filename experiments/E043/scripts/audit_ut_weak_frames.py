"""Audit weak Starlink frames in the author's 250 MS/s supplementary IQ."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy import fft, signal
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, legacy_artifact, require_run_context
from audit_ut_binary_tail import SOURCE, load_samples, sha256
from audit_ut_repeated_data import checked_artifact, read
from code5_core import LOADED, PILOTS, TRAIN_PILOTS, reference_gain, sinc_sample
from educational_fec_demo import encode_block, word_bits
from receiver_models import pn_reference
from ut_receiver_core import capture_grid, decode_variable_pdu, pss_reference


SEED = 2026112143
RATE = 480_000_000
FRAME = RATE // 750
FFT_SIZE = 2048
SLOT = 2112
CP = SLOT - FFT_SIZE
WINDOW = 2048
CFO_BANK = 62_500 * np.arange(-16, 17)
SYNC_GATE = .20
SMOOTH_BINS = 8
PILOT = np.isin(LOADED, PILOTS)
PREFIX = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, dtype=np.uint8)])[:48].astype(float)


def correlation_profile(samples, template):
    conv = signal.fftconvolve(samples, template[::-1].conj(), mode="valid")
    energy = np.r_[0., np.cumsum(abs(samples)**2, dtype=float)]
    power = energy[len(template):] - energy[:-len(template)]
    return abs(conv) / np.sqrt(np.maximum(power * np.vdot(template, template).real, 1e-30))


def norm_match(observed, reference):
    return float(abs(np.vdot(reference, observed)) / max(np.linalg.norm(reference) * np.linalg.norm(observed), 1e-30))


def sync_search(iq, epoch, cfo, pss, sss):
    """Select a PSS peak; SSS is evaluated only at the selected delay and CFO."""
    first = int(round(epoch)) - WINDOW
    length = 2 * WINDOW + 2 * SLOT
    x = np.asarray(iq[first:first + length])
    if first < 0 or len(x) < length:
        raise ValueError("Sync window outside recording")
    indices = np.arange(length)
    best_score = -1.
    for frequency in cfo + CFO_BANK:
        corrected = x * np.exp(-2j*np.pi*frequency*indices/RATE)
        profile = correlation_profile(corrected[:2*WINDOW + SLOT], pss)
        peak = int(np.argmax(profile))
        if profile[peak] > best_score:
            best_score = float(profile[peak])
            best_profile, best_peak = profile, peak
            best_cfo, best_samples = float(frequency), corrected
    fraction = 0.
    if 0 < best_peak < len(best_profile) - 1:
        left, center, right = best_profile[best_peak-1:best_peak+2]**2
        curvature = left - 2*center + right
        if curvature < 0:
            fraction = float(np.clip(.5*(left-right)/curvature, -.5, .5))
    start = best_peak + SLOT + CP
    secondary = norm_match(best_samples[start:start + FFT_SIZE], sss)
    time = first + best_peak + fraction
    return {"pss": best_score, "sss_at_pss": secondary, "epoch_480_samples": time,
            "epoch_residual_480_samples": time - epoch, "cfo_hz": best_cfo,
            "pss_power_db": float(10*np.log10(np.mean(abs(x[best_peak:best_peak+SLOT])**2))),
            "sync_pair_passed": min(best_score, secondary) >= SYNC_GATE}


def body_cp(iq, time, scale, slots, lag_offsets=(-47, 0, 47)):
    indices = time + (SLOT*slots[:, None] + np.arange(8, CP-8))*scale
    prefix = sinc_sample(iq, indices.ravel())
    values = {}
    for delta in lag_offsets:
        tail = sinc_sample(iq, (indices + (FFT_SIZE + delta)*scale).ravel())
        values[str(delta)] = norm_match(tail, prefix)
    return values


def header_diagnostics(values, kind, held_groups):
    header = decode_variable_pdu(values, kind)
    evidence = {k: v for k, v in header.items() if k not in ("bits", "reencoded")}
    if "reencoded" not in header:
        return evidence
    length = len(header["reencoded"])
    recorded = []
    for groups in held_groups:
        mask = np.isin(np.arange(length)//12, groups)
        punctured = values.copy()
        punctured[:length][mask] = 0
        alternate = decode_variable_pdu(punctured, kind)
        row = {"held_groups": groups.tolist(), "held_resources": int(mask.sum()),
               "crc_ok": alternate["crc_ok"], "words_hex": alternate.get("words_hex"),
               "same_words": alternate.get("words_hex") == header["words_hex"]}
        if len(alternate.get("reencoded", [])) == length:
            signs = 1-2*alternate["reencoded"][mask].astype(float)
            test = values[:length][mask].real
            row["held_hard_matches"] = int(np.sum(signs*test > 0))
            row["held_signed_normalized_score"] = float(np.mean(signs*test)/max(np.sqrt(np.mean(test**2)), 1e-30))
        recorded.append(row)
    evidence["punctured_fits"] = recorded
    return evidence


def groups_for_header(tick, kind, length):
    # Keep H1's 48 reference symbols out of holdout. Assign entire contiguous
    # 12-resource groups within each 114-resource word without crossing a word.
    rng = np.random.default_rng(SEED + 2*tick + (kind == "h2"))
    available = [i for i in range(length//12) if (i*12)//114 == (i*12+11)//114]
    if kind == "h1":
        available = [i for i in available if i >= 4]
    count = max(1, len(available)//8)
    order = rng.permutation(available)
    return order[:count], order[count:2*count]


def characterize(iq, tick, found, scale, sss_reference, last_body_slot=300):
    epoch, cfo = found["epoch_480_samples"], found["cfo_hz"]
    spectra, slots = capture_grid(iq, epoch, scale, cfo)
    reference = np.column_stack([pn_reference(int(k), slots[1:]) for k in LOADED])
    impulse = spectra[0]/sss_reference
    channel = np.concatenate([gaussian_filter1d(half, SMOOTH_BINS) for half in np.split(impulse, 2)])
    despread = spectra[1:] * reference.conj()
    pilot_prediction = np.broadcast_to(channel[PILOT], despread[:, PILOT].shape)
    wrong_pn = np.column_stack([pn_reference(int(k), slots[1:]+1) for k in LOADED[PILOT]])
    pilot_wrong = spectra[1:, PILOT] * wrong_pn.conj()
    held_pilots = np.setdiff1d(np.arange(len(PILOTS)), TRAIN_PILOTS)
    body = {}
    eligible = (slots[1:] >= 9) & (slots[1:] <= last_body_slot)
    if eligible.sum() >= 32:
        body["body_slots"] = [int(slots[1:][eligible][0]), int(slots[1:][eligible][-1])]
        body["cp_correlation_by_lag_offset_480_samples"] = body_cp(iq, epoch, scale, slots[1:][eligible])
        body["pilot_coherence"] = {}
        for label, columns in (("interleave_a", TRAIN_PILOTS), ("interleave_b", held_pilots)):
            pred = pilot_prediction[eligible][:, columns].ravel()
            actual = despread[eligible][:, PILOT][:, columns].ravel()
            null = pilot_wrong[eligible][:, columns].ravel()
            body["pilot_coherence"][label] = {"correct_pn": norm_match(actual, pred), "shifted_pn": norm_match(null, pred)}
    # The conjugate-channel product weights soft decisions by received amplitude.
    # One gain from the H1 prefix sets the phase of both headers.
    weighted = (despread[:, ~PILOT] * channel[~PILOT].conj()).ravel()
    g = reference_gain(weighted[:48], PREFIX)
    normalized = weighted/g
    first_values = normalized[:228]
    h1 = header_diagnostics(first_values, "h1", groups_for_header(tick, "h1", len(first_values)))
    result = {"last_slot_observed": int(slots[-1]), "body": body,
              "h1_prefix_match": norm_match(weighted[:48], PREFIX), "h1": h1}
    arrays = {f"tick_{tick}_h1_soft": first_values}
    if h1["crc_ok"] is True:
        first = h1["layout20"]["entries"][0]
        if first["mcs_candidate"] == 5 and first["quantity"] > 0:
            start = 228 + 2048*first["quantity"]
            if start + 114 <= len(normalized):
                values = normalized[start:start+798]
                main_h2 = decode_variable_pdu(values, "h2")
                length = int(main_h2.get("needed_resources", 114))
                result["h2_resource"] = start
                result["h2"] = header_diagnostics(values, "h2", groups_for_header(tick, "h2", length))
                arrays[f"tick_{tick}_h2_soft"] = values
    return result, arrays


def second_grid(iq, first_grid, f_fit, pss, sss, sss_reference):
    # These three first-pass off-grid detections calibrate the timing branch.
    positive_offsets = [(row["tick"], item["epoch_480_samples"])
                        for row in first_grid for item in row["off_grid_controls"] if item["sync_pair_passed"]]
    fit_ticks, measured = np.array(positive_offsets).T
    if len(fit_ticks) < 3 or len(np.unique(fit_ticks)) != len(fit_ticks):
        raise ValueError("Expected separate whole-tick second-grid anchors")
    fit = np.polyfit(fit_ticks, measured - FRAME*fit_ticks, 1)
    scales = 1 + 1e-6*np.arange(-35, 36)
    fits = []
    for tai in measured:
        fits.append([body_cp(iq, tai, scale, np.arange(19, 221, 4), (0,))["0"] for scale in scales])
    scale = float(scales[np.argmax(np.mean(fits, axis=0))])
    records, soft = [], {}
    for tick in range(len(first_grid)):
        estimate = tick*FRAME + float(np.polyval(fit, tick))
        if not WINDOW <= estimate <= len(iq)-2*SLOT-WINDOW-1:
            records.append({"tick": tick, "status": "out_of_capture"})
            continue
        carrier = float(np.polyval(f_fit, tick))
        found = sync_search(iq, estimate, carrier, pss, sss)
        group = "fit_anchor" if tick in fit_ticks else "held"
        row = {"tick": tick, "group": group, "predicted_epoch_480_samples": estimate, "sync": found}
        if found["sync_pair_passed"]:
            # At these slot indices the next first-grid event has not yet started.
            result, arrays = characterize(iq, tick, found, scale, sss_reference, last_body_slot=220)
            row.update(result)
            soft.update({f"offset_grid_{name}": vector for name, vector in arrays.items()})
        records.append(row)
        print("offset", tick, group, round(found["pss"], 3), round(found["sss_at_pss"], 3),
              row.get("h1", {}).get("crc_ok"), row.get("h2", {}).get("crc_ok"), flush=True)
    return {"fit_ticks": fit_ticks.astype(int).tolist(), "epoch_residual_fit": fit.tolist(),
            "sampling_scale": scale, "sampling_scale_grid": scales.tolist(), "cp_calibration_by_frame": fits,
            "records": records}, soft


def main():
    require_run_context()
    inputs = {}
    reference = read(checked_artifact("v022-ut-replay", "E014", "ut_transfer.json", inputs))
    raw = ROOT/SOURCE
    iq = load_samples(raw, reference)
    inputs[raw] = reference["source_sha256"]
    sss_path = legacy_artifact("reference/qin/sssVec.mat")
    inputs[sss_path] = sha256(sss_path)
    if inputs[sss_path] != reference["sss_reference_sha256"]:
        raise ValueError("SSS reference no longer matches E014")
    sss_reference = loadmat(sss_path)["sssVecFull"].ravel()
    frequency = np.zeros(FFT_SIZE, dtype=complex)
    frequency[np.arange(-512, 512) % FFT_SIZE] = fft.fftshift(sss_reference)
    sss = fft.ifft(frequency)
    pss = pss_reference()
    anchors = reference["candidates"]
    initial_epoch = anchors[0]["pss"]["epoch"]
    by_tick = {round((a["pss"]["epoch"]-initial_epoch)/FRAME): a for a in anchors}
    rng = np.random.default_rng(SEED)
    complete = [k for k, a in by_tick.items() if a["complete_frame"]]
    fit_ticks = sorted(rng.permutation(complete)[:5].tolist())
    known_test = sorted(set(by_tick)-set(fit_ticks))
    e_fit = np.polyfit(fit_ticks, [by_tick[k]["pss"]["epoch"]-k*FRAME for k in fit_ticks], 1)
    f_fit = np.polyfit(fit_ticks, [by_tick[k]["cfo_hz"] for k in fit_ticks], 1)
    scale = float(np.median([by_tick[k]["sampling_scale"] for k in fit_ticks]))
    end_tick = int((len(iq)-initial_epoch-2*SLOT-WINDOW)//FRAME)
    records, arrays = [], {}
    for tick in range(end_tick+1):
        estimate = FRAME*tick + float(np.polyval(e_fit, tick))
        carrier = float(np.polyval(f_fit, tick))
        found = sync_search(iq, estimate, carrier, pss, sss)
        offsets = []
        for low, high in ((-620, -420), (-320, -100), (100, 320), (420, 620)):
            offset = float(rng.uniform(low, high))
            moved = estimate + offset*RATE/1e6
            if WINDOW <= moved <= len(iq)-2*SLOT-WINDOW-1:
                offsets.append({"offset_us": offset, **sync_search(iq, moved, carrier, pss, sss)})
        group = "fit_anchor" if tick in fit_ticks else "held_anchor" if tick in by_tick else "previously_unlisted"
        row = {"tick": tick, "group": group, "predicted_epoch_480_samples": estimate,
               "predicted_cfo_hz": carrier, "sync": found, "off_grid_controls": offsets}
        if found["sync_pair_passed"]:
            result, soft = characterize(iq, tick, found, scale, sss_reference[LOADED % 1024])
            row.update(result)
            arrays.update(soft)
        records.append(row)
        print(tick, group, round(found["pss"], 3), round(found["sss_at_pss"], 3),
              row.get("h1", {}).get("crc_ok"), row.get("h2", {}).get("crc_ok"), flush=True)
    offset_grid, offset_soft = second_grid(iq, records, f_fit, pss, sss, sss_reference[LOADED % 1024])
    arrays.update(offset_soft)
    report = {
        "schema": "starlink-ut-weak-frames/v1",
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
        "seed": SEED, "capture_rate_hz": 250_000_000, "analysis_rate_hz": RATE,
        "nominal_frame_rate_hz": RATE//FRAME, "fit_ticks": fit_ticks, "held_anchor_ticks": known_test,
        "epoch_residual_fit": e_fit.tolist(), "cfo_fit_hz": f_fit.tolist(), "sampling_scale": scale,
        "pss_window_half_width_480_samples": WINDOW, "cfo_bank_hz": CFO_BANK.tolist(),
        "joint_threshold": SYNC_GATE, "sss_channel_gaussian_sigma_bins": SMOOTH_BINS,
        "records": records, "offset_grid": offset_grid,
        "scope": "Same 50 ms capture as E014. Search conditional on its timing/frequency region. CRC is post-decision, with weak headers subject to punctured rechecks.",
    }
    output = experiment_output("E043")
    output.mkdir(parents=True, exist_ok=True)
    (output/"ut_weak_frames.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output/"ut_weak_headers.npz", **arrays)


if __name__ == "__main__":
    main()
