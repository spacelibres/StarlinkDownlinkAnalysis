"""Read finite constraints immediately before the frozen repeated byte anchor."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_qin_low_codes import qam_words
from audit_qin_mcs6_transfer import ANCHOR, positions_of
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


FORMATS = (6, 35, 42, 93)
REVERSE = tuple(int(f"{value:08b}"[::-1], 2) for value in range(256))


def prefix_observations(source, saved, witness):
    observed = positions_of(source, ANCHOR)
    if observed != saved["records"]["anchor"]["all_exact_starts"]:
        raise ValueError("E038 hard anchor positions disagree")
    verified = [r["start_byte"] for r in witness["frozen"]["hits"] if r["reference_bit"] == r["transfer"]]
    rows = []
    for i in sorted(set(observed + verified)):
        if i < 8:
            raise ValueError("Encountered an anchor without eight prior bytes")
        prefix = source[i-8:i]
        rows.append({"anchor_byte": i, "prefix_hex": prefix.hex(), "first_byte_is_08": prefix[0] == 0x08,
                     "penultimate_byte_is_zero": prefix[-2] == 0, "low_three_is_010": prefix[-1] & 7 == 2,
                     "receiver_order_value": prefix[-1] >> 3, "reversed_order_value": REVERSE[prefix[-1]] & 31,
                     "anchor_reference_assisted": i not in observed})
    transitions = []
    for left, right in zip(rows, rows[1:]):
        raw_step = (right["receiver_order_value"] - left["receiver_order_value"]) % 32
        reverse_step = (right["reversed_order_value"] - left["reversed_order_value"]) % 32
        transitions.append({"anchor_bytes": [left["anchor_byte"], right["anchor_byte"]],
                            "same_first_seven": left["prefix_hex"][:14] == right["prefix_hex"][:14],
                            "receiver_order_increment": raw_step, "reversed_order_increment": reverse_step,
                            "anchor_reference_assisted": left["anchor_reference_assisted"] or right["anchor_reference_assisted"]})
    return {"prefixes": rows, "transitions": transitions}


def main():
    require_run_context()
    inputs = {}
    audit = read(checked_artifact("v034-qin-low-mcs-audit", "E036", "qin_low_mcs_audit.json", inputs))
    symbols = checked_artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_symbols.npz", inputs)
    transfer = read(checked_artifact("v036-qin-mcs6-transfer", "E038", "qin_mcs6_transfer.json", inputs))
    witness = read(checked_artifact("v036-qin-mcs6-anchor-edge", "E038", "qin_mcs6_anchor_edge.json", inputs))
    prior = read(checked_artifact("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json", inputs))
    mapping = prior["cross32_mapping"]
    points32 = np.array([complex(p["i"], p["q"]) for p in mapping])/np.sqrt(20)
    labels32 = labels_as_bits(np.array([p["label"] for p in mapping]), 5)
    pn = equality_predictions(np.arange(16384), pn_bits(), audit["pn_model"])
    stats = {}
    records = []
    with np.load(symbols, allow_pickle=False) as a:
        for code in FORMATS:
            k = audit["formats"][str(code)]["candidate_boundary"]
            b = qam_words(a[f"mcs{code}_symbols"], code, points32, labels32)[:, :k]
            data = np.packbits(b ^ pn[:k], axis=1, bitorder="big")
            counts = [len(positions_of(row.tobytes(), ANCHOR)) for row in data]
            stats[code] = {"tested_words": len(data), "matched_words": sum(n > 0 for n in counts),
                           "anchor_counts_by_word": counts, "searched_bytes_per_word": k//8}
            if code != 6:
                continue
            for bits, frame, earlier, masked in zip(data, a["mcs6_frames"], transfer["words"], witness["words"], strict=True):
                if frame != earlier["frame"] or earlier["word"] != masked["word"]:
                    raise ValueError("Archive word ordering changed")
                records.append({"frame": int(frame), "word": earlier["word"], "calibration": earlier["calibration"],
                                **prefix_observations(bits.tobytes(), earlier, masked)})
    result = {"schema": "starlink-qin-anchor-prefix/v1", "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()},
              "group_seed": audit["whole_group_seed"], "groups": audit["formats"]["6"]["groups"],
              "anchor_hex": ANCHOR.hex(), "format_search": stats, "mcs6_words": records}
    out = experiment_output("E039")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qin_anchor_prefix.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print([(c, s["matched_words"], s["tested_words"]) for c, s in stats.items()], flush=True)


if __name__ == "__main__":
    main()
