"""Test whether empirical tessellation diversity survives a change to analytic PN coordinates."""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_narrowband_cycle import QIN, code_indices, permute_lanes
from audit_qin_exemplars import remove_header_gain
from audit_qin_lengths import pdu_stream
from audit_residual_transfer import coherence
from audit_ut_binary_tail import sha256
from audit_ut_repeated_data import checked_artifact, read
from audit_ut_weak_frames import PREFIX
from audit_wav_tessellation import forward
from layout_core import NONPILOT
from receiver_models import pn_reference


SEED = 2026112149
SIZE = 60
GROUPS = 16
MIN_PURITY = .95
MAX_GROUP_MISMATCH = .02


@lru_cache(maxsize=None)
def canonical(value):
    """Binary cyclic-equivalence key modulo one common sign."""
    inverse = bytes(v ^ 1 for v in value)
    bits = min(s[i:]+s[:i] for s in (value, inverse) for i in range(SIZE))
    return np.packbits(np.frombuffer(bits, dtype=np.uint8)).tobytes().hex()


def canon(bits):
    return canonical(np.asarray(bits, dtype=np.uint8).tobytes())


def unpack(key):
    return 1-2*np.unpackbits(np.frombuffer(bytes.fromhex(key), dtype=np.uint8))[:SIZE].astype(int)


def signal_slots(grid, a):
    data = grid[:, :SIZE*GROUPS].reshape(len(grid), GROUPS, SIZE)
    samples = data[:, a]
    phase_purity = (abs(samples.imag) <= .14*abs(samples)).mean(axis=(1, 2))
    reference = samples.real.mean(axis=1) < 0
    agreement = ((samples.real < 0) == reference[:, None]).mean(axis=(1, 2))
    balance = reference.sum(axis=1)
    eligible = (phase_purity >= MIN_PURITY) & (agreement >= 1-MAX_GROUP_MISMATCH) & (balance >= 6) & (balance <= 54)
    return data, phase_purity, agreement, reference, eligible


def collect_words(paths, names, pn, pn_wrong, rot, rng):
    spectral_order = rng.permutation(GROUPS)
    calibration, held = spectral_order[:8], spectral_order[8:]
    calibration_groups = rng.permutation(names)[:len(names)//2].tolist()
    candidates, counts = [], {}
    for frame, group, stored, flat in pdu_stream(paths, names, pn, PREFIX):
        direct = flat.reshape(300, len(NONPILOT))
        projected, purity, agreement, proposed, eligible = signal_slots(direct, calibration)
        wrong = remove_header_gain(stored*pn_wrong, PREFIX).reshape(300, len(NONPILOT))
        _, _, _, _, wrong_eligible = signal_slots(wrong, calibration)
        qin = (direct*rot.conj())[:, :SIZE*GROUPS].reshape(300, GROUPS, SIZE)
        rec = []
        for slot in np.flatnonzero(eligible):
            first = int(slot)*len(NONPILOT)%SIZE
            signed = np.roll(proposed[slot], first)
            qin_tested = qin[slot, held].real
            observed_qin = np.roll(qin_tested.mean(axis=0) < 0, first)
            rec.append({
                "frame": frame, "group": group, "slot": int(slot)+2, "calibration_phase_purity": float(purity[slot]),
                "calibration_repeat_agreement": float(agreement[slot]), "analytic_family": canon(signed),
                "held_sign_differences_from_slot_fit": int(
                    np.count_nonzero((projected[slot, held].real < 0) != proposed[slot])),
                "held_frequency_phase_purity": float(np.mean(abs(projected[slot, held].imag) <= .14*abs(projected[slot, held]))),
                "qin_measured_held_family": canon(observed_qin),
                "calibration_word_bits": signed,
                "held_analytic_values": projected[slot, held].real.copy(),
                "held_qin_values": qin_tested.copy(),
            })
        counts[frame] = {"group": group, "eligible_analytic_slots": int(eligible.sum()),
                         "wrong_slot_pn_eligible_slots": int(wrong_eligible.sum())}
        candidates.extend(rec)
    training = Counter(r["analytic_family"] for r in candidates if r["group"] in calibration_groups)
    winner = training.most_common(1)[0][0]
    return candidates, counts, calibration_groups, spectral_order, winner


def archive_transfer(rows, counts, calibration, groups, word, rotation, boundaries):
    phases = np.array([np.roll(word, i) for i in range(SIZE)])
    bank = np.concatenate([phases, -phases], axis=0)
    total = Counter()
    families = Counter()
    records = []
    for row in rows:
        slot = row["slot"]-2
        coordinate = (slot*len(NONPILOT)+np.arange(SIZE)) % SIZE
        source = 1-2*row["calibration_word_bits"].astype(int)
        best = int(np.argmax(bank @ source))
        chosen = bank[best, coordinate]
        truth = row["held_analytic_values"]
        empirical_truth = row["held_qin_values"]
        difference = (truth < 0) != (chosen < 0)
        converted = chosen*rotation[slot, :SIZE]
        empirical_difference = (empirical_truth < 0) != (converted.real < 0)
        partition = "training" if row["group"] in calibration else "held"
        boundary = boundaries[row["frame"]]
        end = boundary["nominal"]["phase_refined_edge"]
        start = slot*len(NONPILOT)
        region = "entire_slot_before" if start+len(NONPILOT) <= end else "entire_slot_after" if start >= end else "straddles"
        total[partition] += 1
        key = canon(np.roll(converted.real < 0, start % SIZE))
        families[row["qin_measured_held_family"]] += 1
        serial = {k: v for k, v in row.items() if k not in ("calibration_word_bits", "held_analytic_values", "held_qin_values")}
        records.append({**serial, "partition": partition, "selected_negative": bool(best >= SIZE),
                        "selected_phase_subtracted": best % SIZE, "boundary_relation": region,
                        "phase_minus_measured_boundary_mod60": int((best-end) % SIZE),
                        "phase_minus_header_predicted_boundary_mod60": int((best-boundary["per_word_end"]) % SIZE),
                        "analytic_held_prediction_differences": int(difference.sum()),
                        "qin_held_prediction_differences": int(empirical_difference.sum()),
                        "transformed_qin_family": key})
    return {"training_groups": calibration, "held_groups": sorted(set(groups)-set(calibration)),
            "slots_by_partition": dict(total), "per_frame_candidate_counts": counts,
            "analytic_families": dict(Counter(r["analytic_family"] for r in rows)),
            "qin_families": dict(families), "slots": records}


def wav_check(y, h, e042, mask, template):
    train, held = [e042["groups"][name] for name in ("training", "held")]
    u, v = [y[group].mean(axis=0).ravel() for group in (train, held)]
    design = forward(h, e042["fitted_offset_hz"]).reshape(len(mask), 4, 15)
    resources = np.asarray(e042["reference_comparison"]["resource_index_by_lane_and_slot_class"])
    all_rows = []
    for shift in range(SIZE):
        model = np.einsum("jck,ck->jc", design, template[(resources-shift) % SIZE])
        case = []
        for side in (0, 1):
            target = np.repeat(np.arange(2) == side, h.shape[1])
            training, reserved = target & ~mask, target & mask
            w = np.linalg.lstsq(model[training], u[training], rcond=1e-8)[0]
            nmse = np.linalg.norm(model[training] @ w-u[training])**2/np.linalg.norm(u[training])**2
            case.append({"training_nmse": float(nmse), "held_score": coherence(model[reserved] @ w, v[reserved])})
        all_rows.append(case)
    selected = [int(np.argmin([c[s]["training_nmse"] for c in all_rows])) for s in (0, 1)]
    return {"phase_by_side_subtracted": selected, "fits": all_rows}


def npz_carrier_prediction(record, data, word, alternatives, rng):
    references = np.stack([word, *alternatives, *[permute_lanes(word, rng) for _ in range(199)]])
    cases = {}
    for guard, branches in record["frontends"].items():
        cases[guard] = {}
        for scheme, windows in branches.items():
            detail = {}
            grouped_null = []
            for row in windows:
                name = row["window"]
                pos = (np.arange(SIZE)[:, None, None]+1004*np.arange(15)[None, :, None]
                       +code_indices(row["carriers"])[None, None, :]) % SIZE
                waves = references[:, pos].astype(float)
                waves -= waves.mean(axis=2, keepdims=True)
                waves /= np.maximum(np.linalg.norm(waves, axis=2, keepdims=True), 1e-30)
                folds, scores = [], []
                for i in (0, 1):
                    x, y = data[f"{guard}_{scheme}_{name}_fold{i}"]
                    fit = np.einsum("tjsc,fsc->tfjc", waves, x)
                    power = abs(fit)**2
                    channels = np.arange(len(row["carriers"]))
                    case, cscore = [], []
                    for carrier in channels:
                        best = power[..., channels != carrier].sum(axis=-1).argmax(axis=-1)
                        index = (np.arange(len(waves))[:, None], np.arange(len(x))[None, :], best)
                        pred = fit[index][..., carrier, None]*waves[np.arange(len(waves))[:, None], best, :, carrier]
                        target = y[..., carrier]
                        score = np.einsum("tfs,fs->t", pred.conj(), target).real/np.maximum(
                            np.sqrt(np.sum(abs(pred)**2, axis=(1, 2)))*np.linalg.norm(target), 1e-30)
                        cscore.append(score)
                        case.append({"carrier": row["carriers"][carrier], "target_score": float(score[0]),
                                     "winning_phases_from_other_carriers": best[0].tolist()})
                    scores.extend(cscore)
                    folds.append(case)
                stats = np.mean(scores, axis=0)
                grouped_null.append(stats[1:])
                detail[name] = {"folds": folds, "held_carrier_score": float(stats[0]),
                                "wrong_reference_family_scores": stats[1:1+len(alternatives)].tolist(),
                                "shuffled_words": stats[1+len(alternatives):].tolist()}
            family = np.max(grouped_null, axis=0)
            for row in detail.values():
                row["family_control_max"] = float(max(family))
                row["family_controls_ge_observed"] = int(np.count_nonzero(family >= row["held_carrier_score"]))
            cases[guard][scheme] = detail
    return cases


def main():
    require_run_context()
    inputs = {}
    def artifact(run, experiment, filename):
        return checked_artifact(run, experiment, filename, inputs)
    rng = np.random.default_rng(SEED)
    snap = "v032-qin-snapshot"
    source = read(artifact(snap, "E034", "qin_snapshot_source.json"))
    names = source["hard_groups"]
    paths = [artifact(snap, "E034", f"qin_hard_{n}.npz") for n in names]
    p = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    wrong_p = np.column_stack([pn_reference(int(k), np.arange(3, 303)) for k in NONPILOT]).conj()
    inputs[QIN] = sha256(QIN)
    qin = loadmat(QIN)["referenceTemplateRotations"]
    conversion = np.exp(.5j*np.pi*qin[NONPILOT % 1024, 1:].T)*p
    rows, counts, train_groups, permutation, dominant = collect_words(paths, names, p, wrong_p, conversion, rng)
    prior = read(artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json"))
    gold = np.asarray(prior["cycle_folds"][0]["signed_template"])
    matches_e025 = canon(gold < 0) == dominant
    word = gold if matches_e025 else unpack(dominant)
    boundaries = read(artifact("v033-qin-lengths-final", "E035", "qin_lengths.json"))
    frames = {r["frame"]: r for r in boundaries["frames"]}
    wide = archive_transfer(rows, counts, train_groups, names, word, conversion, frames)
    others = [unpack(key) for key in sorted(wide["qin_families"]) if len(set(unpack(key))) > 1]
    wav_config = read(artifact("v040-wav-tessellation", "E042", "wav_tessellation.json"))
    with np.load(artifact("v040-wav-tessellation", "E042", "wav_tessellation.npz"), allow_pickle=False) as s, np.load(
            artifact("v039-wav-joint-phase", "E041", "wav_joint_phase.npz"), allow_pickle=False) as w:
        wav = wav_check(w["mode_0_source_0_aligned_spectra"], w["harmonics"], wav_config, s["held_harmonic_mask"], word)
    stage = "v046-cross-capture-phase-final"
    npz_meta = read(artifact(stage, "E048", "residual_transfer.json"))["npz"]
    with np.load(artifact(stage, "E048", "residual_transfer.npz"), allow_pickle=False) as d:
        npz = npz_carrier_prediction(npz_meta, d, word, others, rng)
    report = {"schema": "starlink-reference-word-families/v1", "seed": SEED,
               "inputs_sha256": {k.relative_to(ROOT).as_posix(): v for k, v in inputs.items()},
               "analytic_training_family": dominant, "matches_e025_family": matches_e025,
               "spectral_split_60": permutation.tolist(),
               "wideband": wide, "wav_phase": wav, "npz_hold_one_carrier": npz,
               "scope": "Conditional on the gated slots in the local 117-frame subset. Narrow modes are reference-assisted."}
    output = experiment_output("E049")
    output.mkdir(parents=True, exist_ok=True)
    (output/"reference_word_families.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Archive slots", wide["slots_by_partition"], "analytic families", len(wide["analytic_families"]),
          "Qin families", len(wide["qin_families"]), "WAV phase", wav["phase_by_side_subtracted"], flush=True)


if __name__ == "__main__":
    main()
