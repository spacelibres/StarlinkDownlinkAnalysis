"""Compare five and six families on the same measured MCS 85 soft input."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross_rate_checks import shift_systematic
from audit_cross32_decoded_counters import pair_fields
from audit_cross32_partial_decode import BIT_LENGTH, decode, raw_inputs, raw_record_metrics
from audit_mcs66_partial_decoder import output_decisions, pn_metrics, syndrome_count
from audit_mcs85_records import system_bytes
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


EDGE = 12288


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model_run = "v125-cross-rate-checks"
    with np.load(art(model_run, "E127", "mcs66_six_families.npz"), allow_pickle=False) as z:
        six = z["row_support"]
    with np.load(art("v085-mcs66-window-check", "E085", "mcs66_phase_window_check.npz"), allow_pickle=False) as z:
        five = z["row_support"]
    if not np.array_equal(six[:len(five)], np.sort(five, axis=1)) or len(six) != len(five) + 512:
        raise ValueError("The five-family prefix or added family differs")
    fake = np.vstack((five, shift_systematic(six[len(five):], EDGE)))
    labels = read(art("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    original = art("v028-ut-cross32-extract-02", "E030", "ut_cross32_symbols.npz")
    with np.load(original, allow_pickle=False) as y:
        frames, prior, bits, known = raw_inputs(85, y, labels, EDGE)
    original_mask = np.ones_like(bits)
    prior_run = "v123-cross32-partial-decode"
    earlier = read(art(prior_run, "E125", "cross32_partial_decode.json"))["formats"]["85"]["raw250"]
    setting = earlier["chosen"]
    held = np.isin(frames, labels["frame_split"]["reserved"])
    pn = equality_predictions(np.arange(BIT_LENGTH), pn_bits(), labels["frozen_pn_model"]).astype(bool)
    metrics, decoded = {}, {}
    for name, support in (("five", five), ("six", six), ("five_plus_shifted_sixth", fake)):
        posterior = decode(prior, support, *setting)
        values, mask = output_decisions(posterior, bits, original_mask)
        decoded[name] = values
        metrics[name] = pn_metrics(values[held], mask[held], original_mask[held], pn, known[held], frames[held])
        metrics[name]["violations_under_six_true_families"] = int(syndrome_count(values[held], six).sum())
    with np.load(art(prior_run, "E125", "cross32_partial_decisions.npz"), allow_pickle=False) as z:
        prior_bits = np.unpackbits(z["85_decoded_bits"], axis=1)
        if not np.array_equal(prior_bits, decoded["five"]):
            raise ValueError("Five-family decoder did not exactly replay E125")
    record = read(art("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    if record["frames_split"] != labels["frame_split"] or record["pn_model"] != labels["frozen_pn_model"]:
        raise ValueError("Record scoring split or PN disagrees")
    record_scores = raw_record_metrics(record, frames, bits, decoded, pn, EDGE)
    pairs = []
    for frame in record["frame_results"]:
        number = frame["frame"]
        if number not in record["frames_split"]["reserved"]:
            continue
        streams = {name: system_bytes(y[frames == number], EDGE, pn).ravel() for name, y in decoded.items()}
        pairs.extend(pair_fields(frame, streams, record["fitted_distances_by_exact_label"]))
    counter_scores = {
        name: {"plus1": sum(r["observations"][name]["step"] == 1 for r in pairs),
               "plus1_to_4": sum(1 <= r["observations"][name]["step"] <= 4 for r in pairs),
               "both_tags": sum(all(e["tag"] == r["tag"] for e in r["observations"][name]["ends"]) for r in pairs)}
        for name in decoded
    }
    report = {"schema": "starlink-six-family-mcs85-replay/v1", "fixed_decoder": setting, "held_pn": metrics,
              "held_frozen_record_scores": record_scores, "held_frozen_pairs": pairs,
              "held_counter_scores": counter_scores, "raw_groups": labels["frame_split"],
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()}}
    out = experiment_output("E127")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "mcs85_six_family_decisions.npz",
                        frames=frames, **{k: np.packbits(v, axis=1) for k, v in decoded.items()})
    (out / "mcs85_six_family_replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                                    encoding="utf-8")
    print("Held PN errors:", {k: v["observed_errors"] for k, v in metrics.items()})
    print("Frozen counter pairs:", len(pairs), counter_scores)


if __name__ == "__main__":
    main()
