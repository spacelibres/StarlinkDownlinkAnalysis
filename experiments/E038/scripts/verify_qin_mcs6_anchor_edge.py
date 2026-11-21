"""Test the Code 5 phase prediction on an edge bit held out from anchor matching."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_qam_pn15 import equality_predictions
from audit_qin_mcs6_structure import K
from audit_qin_mcs6_transfer import ANCHOR
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits


def scan(word, carrier, q, motif):
    hard = (word[:K].real < 0).astype(np.uint8) ^ q
    hits = []
    windows = 0
    for i, address in enumerate(carrier["coordinates"]):
        if address >= K or carrier["known_reference"][i] != -1:
            continue
        first_byte = (max(0, address - len(motif) + 1)+7)//8
        last_byte = min((K-len(motif))//8, address//8)
        for byte in range(first_byte, last_byte+1):
            start = byte*8
            omit = address-start
            mask = np.arange(len(motif)) != omit
            windows += 1
            if not np.array_equal(hard[start:start+len(motif)][mask], motif[mask]):
                continue
            expected = int(motif[omit] ^ q[address])
            transfer = carrier["exact_erasure"]
            hits.append({"start_byte": byte, "erased_word_coordinate": address,
                         "reference_bit": expected, "raw_hard": carrier["raw_hard"][i],
                         "transfer": transfer["estimated"][i] if "estimated" in transfer else -1,
                         "wrong_half_turn": transfer["wrong_half_turn"][i] if "wrong_half_turn" in transfer else -1})
    return {"tested_windows": windows, "hits": hits}


def main():
    require_run_context()
    inputs = {}
    audit = read(checked_artifact("v034-qin-low-mcs-audit", "E036", "qin_low_mcs_audit.json", inputs))
    path = checked_artifact("v034-qin-low-mcs-extract", "E036", "qin_low_mcs_symbols.npz", inputs)
    transfer = read(checked_artifact("v036-qin-mcs6-transfer", "E038", "qin_mcs6_transfer.json", inputs))
    model = audit["pn_model"]
    p = equality_predictions(np.arange(K), pn_bits(), model)
    wrong_p = equality_predictions(np.arange(K), pn_bits(), {**model, "phase": model["phase"]+1})
    motif = np.unpackbits(np.frombuffer(ANCHOR, dtype=np.uint8))
    rows = []
    with np.load(path, allow_pickle=False) as data:
        for row, observed, frame in zip(transfer["words"], data["mcs6_symbols"], data["mcs6_frames"], strict=True):
            if row["frame"] != frame:
                raise ValueError("Transfer and E036 frame indexes do not align")
            carrier = row["carrier_transfer"]
            rows.append({"frame": int(frame), "word": row["word"], "calibration": row["calibration"],
                         "preexisting_exact_starts": row["records"]["anchor"]["all_exact_starts"],
                         "frozen": scan(observed, carrier, p, motif),
                         "wrong_phase": scan(observed, carrier, wrong_p, motif),
                         "reversed_motif": scan(observed, carrier, p, motif[::-1])})
    out = experiment_output("E038")
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema": "starlink-qin-mcs6-anchor-edge/v1", "words": rows,
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in inputs.items()},
              "scope": "Each search window requires 95 hard matches and omits the one k=511 coordinate. "
                       "The anchor and Code 5 gain are frozen. One retrospective positive is a conditional check."}
    (out / "qin_mcs6_anchor_edge.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(sum(len(row["frozen"]["hits"]) for row in rows), "target payload edge anchors", flush=True)


if __name__ == "__main__":
    main()
