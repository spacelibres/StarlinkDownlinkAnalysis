"""Replay fixed record and protocol comparisons after full channel decoding."""

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_cross32_decoded_counters import pair_fields
from audit_mcs85_record_boundary import protocol_checks
from audit_mcs85_records import system_bytes
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_full_code import SYSTEMATIC


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    model_run = "v126-mcs66-full-validation"
    path = art(model_run, "E128", "mcs66_full_decoder_words.npz")
    with np.load(path, allow_pickle=False) as z:
        bits = np.unpackbits(z["mcs85_ordinary_bits"], axis=1)
        frames = z["mcs85_frames"]
        if not z["mcs85_ordinary_pass"].all() or not np.array_equal(z["mcs85_ordinary_bits"], z["mcs85_masked_bits"]):
            raise ValueError("Record pass requires matching ordinary and hidden-symbol codewords")
    model = read(art("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    pn = equality_predictions(np.arange(SYSTEMATIC), pn_bits(), model["pn_model"])
    records, pairs, frame_reports = [], [], []
    aggregate = Counter()
    for r in model["frame_results"]:
        frame = r["frame"]
        if frame not in model["frames_split"]["reserved"]:
            continue
        data = system_bytes(bits[frames == frame], SYSTEMATIC, pn).ravel()
        eligible = [row for row in r["length_predictions"] if row["status"] == "matched"]
        records.extend({"frame": frame, **row, "prefix_hex": data[row["reference"]:row["reference"] + 20].tobytes().hex()}
                       for row in eligible)
        checks = protocol_checks(data, r)
        for c in checks:
            for order in ("lsb", "msb"):
                aggregate.update({k: len(v) if isinstance(v, list) else v for k, v in c[order].items()})
        frame_reports.append({"frame": frame, "checks": checks})
        pairs.extend(pair_fields(r, {"full": data}, model["fitted_distances_by_exact_label"]))
    offsets = list(range(4)) + list(range(8, 20))
    words = [bytes.fromhex(i["prefix_hex"]) for i in records]
    output = {
        "schema": "starlink-full-mcs85-record-replay/v1", "records": records, "pairs": pairs,
        "prefix_bytes": {i: dict(Counter(f"{word[i]:02x}" for word in words)) for i in offsets},
        "pair_steps": dict(Counter(r["observations"]["full"]["step"] for r in pairs)),
        "both_tags_match": sum(all(x["tag"] == r["tag"] for x in r["observations"]["full"]["ends"]) for r in pairs),
        "aggregate_protocol": aggregate, "frame_reports": frame_reports,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in inputs.items()},
    }
    out = experiment_output("E128")
    out.mkdir(parents=True, exist_ok=True)
    (out / "mcs85_full_records.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Complete record views", len(records), "fixed counter pairs", len(pairs))
    print("Protocol comparisons", dict(aggregate))


if __name__ == "__main__":
    main()
