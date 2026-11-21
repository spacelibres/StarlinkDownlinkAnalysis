"""Read every Qin exemplar header using the receiver frozen by E034."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qin_exemplars import frame_headers, remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from snapshot_qin_exemplars import ArchiveReader, ETAG, SIZE, URL, as_complex
from ut_receiver_core import decode_variable_pdu

FIRST_WIDTH = 228
SECOND_WINDOW = 798
RESOURCE_PER_SLOT = len(NONPILOT)
SLOTS = np.arange(2, 302)


def compact(header):
    return {k: v for k, v in header.items() if k not in ("bits", "reencoded")}


def scan_headers(dataset, toa, sinr):
    p = np.column_stack([pn_reference(int(k), SLOTS) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    h1_in = np.zeros((dataset.shape[0], FIRST_WIDTH), dtype=np.complex128)
    h2_in = np.zeros((dataset.shape[0], SECOND_WINDOW), dtype=np.complex128)
    h2_length = np.zeros(dataset.shape[0], dtype=np.int32)
    h2_start = np.full(dataset.shape[0], -1, dtype=np.int32)
    rows = []
    for start in range(0, dataset.shape[0], dataset.chunks[0]):
        stop = min(start+dataset.chunks[0], dataset.shape[0])
        # Only the two header spans are published. The leading 24 slots usually cover both.
        early = as_complex(dataset[start:stop, 1:25, :])[..., NONPILOT % 1024]
        for index, window in enumerate(early, start):
            flat = remove_header_gain(window * p[:len(window)], signs)
            h1, h2, second = frame_headers(flat)
            h1_in[index] = window.ravel()[:FIRST_WIDTH]
            if second is not None and second < len(SLOTS)*RESOURCE_PER_SLOT:
                end = min(second + SECOND_WINDOW, len(SLOTS)*RESOURCE_PER_SLOT)
                if end > len(flat):
                    slot_count = (end+RESOURCE_PER_SLOT-1)//RESOURCE_PER_SLOT
                    window = as_complex(dataset[index, 1:1+slot_count, :])[:, NONPILOT % 1024]
                    flat = remove_header_gain(window * p[:slot_count], signs)
                    h2 = decode_variable_pdu(flat[second:end], "h2") if end-second >= 114 else {}
                h2_in[index, :end-second] = window.ravel()[second:end]
                h2_start[index] = second
                h2_length[index] = end-second
            rows.append({"frame": index, "toa_s": float(toa[index]), "sinr_db": float(sinr[index]),
                         "h1": compact(h1), "h2": compact(h2), "h2_start": second})
        if start % 130 == 0 or stop == dataset.shape[0]:
            print(f"Scanned {stop}/{dataset.shape[0]} exemplar headers", flush=True)
    return rows, {"h1_symbols": h1_in, "h2_symbols": h2_in,
                  "h2_start": h2_start, "h2_length": h2_length,
                  "frame_index": np.arange(len(rows)), "toa_s": toa, "sinr_db": sinr}


def main():
    require_run_context()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unverified-source", action="store_true")
    args = parser.parse_args()
    cache = ROOT / ".cache" / "qin-mat-pages"
    cache.mkdir(parents=True, exist_ok=True)
    old_inputs = {}
    prior = read(checked_artifact("v032-qin-exemplar-audit", "E034", "qin_exemplar_audit.json", old_inputs))
    reader = ArchiveReader(cache, args.allow_unverified_source)
    with reader, h5py.File(reader, "r") as mat:
        hard = mat["yDataDec"]
        if hard.shape != (1009, 301, 1024):
            raise ValueError("The archive dimensions have changed")
        rows, arrays = scan_headers(hard, mat["TOAs"][...].ravel(), mat["SINRs"][...].ravel())
    overlap = [row["frame"] for row in prior["headers"] if rows[row["frame"]] == row]
    if len(overlap) != len(prior["headers"]):
        raise ValueError("Head replay changed an already archived header")
    histogram = Counter(x["mcs_candidate"] for row in rows if row["h2"].get("crc_ok") is True
                        and row["h2"]["layout20"]["packing_valid"] for x in row["h2"]["layout20"]["entries"])
    report = {"schema": "starlink-qin-header-survey/v1", "frames": rows, "full_overlap_replay": overlap,
              "crc_accepted_h1": sum(row["h1"].get("crc_ok") is True for row in rows),
              "crc_accepted_h2": sum(row["h2"].get("crc_ok") is True for row in rows),
              "crc_and_packing_accepted_h2": sum(row["h2"].get("crc_ok") is True
                                                  and row["h2"]["layout20"]["packing_valid"] for row in rows),
              "mcs_descriptor_counts": dict(sorted(histogram.items())), "archive": {"url": URL, "etag": ETAG, "bytes": SIZE,
              "local_range_cache_transport": "The original range cache used an unverified TLS connection.",
              "network_tls_validated": not args.allow_unverified_source,
              "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
              "inputs_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in old_inputs.items()},
              "scope": "Conditional 20-bit layout on CRC-passing headers from one named RF data archive."
                       " Header inventory does not validate each payload's modulation or encoder."}
    out = experiment_output("E036")
    out.mkdir(parents=True, exist_ok=True)
    (out / "qin_header_survey.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out / "qin_header_symbols.npz", **arrays)
    print("Accepted H1/H2/packing", report["crc_accepted_h1"], report["crc_accepted_h2"],
          report["crc_and_packing_accepted_h2"], "of", len(rows), flush=True)
    print("Descriptor counts", dict(sorted(histogram.items())), flush=True)


if __name__ == "__main__":
    main()
