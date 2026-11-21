"""Freeze the authors' measured TOA and Doppler columns from two pinned MAT pages."""

import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import PROJECT_ROOT, experiment_output, require_run_context
from audit_full_tail_phase import CacheOnly
from audit_ut_repeated_data import checked_artifact, read
from snapshot_qin_exemplars import ETAG, SIZE, URL


def main():
    require_run_context()
    inputs = {}
    source = read(checked_artifact("v032-qin-snapshot", "E034", "qin_snapshot_source.json", inputs))
    headers = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))["frames"]
    known = {tuple(r["bytes"]): r["sha256"] for r in source["pages"]}
    reader = CacheOnly(PROJECT_ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as data:
        toa = data["TOAs"][...].ravel()
        beta = data["betas"][...].ravel()
        filename = "".join(map(chr, data["filename"][...].ravel()))
    if filename != source["upstream_capture_name"] or not np.array_equal(toa, [r["toa_s"] for r in headers]):
        raise ValueError("Archive filename or TOA column changed")
    pages = [reader.receipts[i] for i in sorted(reader.receipts)]
    if any(known[tuple(r["bytes"])] != r["sha256"] for r in pages) or not np.isfinite(beta).all():
        raise ValueError("Archive source bytes or Doppler column changed")
    out = experiment_output("E095")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "qin_time_doppler.npz", frame=np.arange(len(toa)), toa_s=toa, beta=beta)
    receipt = {"schema": "starlink-qin-time-doppler/v1", "source": URL, "etag": ETAG, "bytes": SIZE,
               "archive_provenance": source["cache_transport_caveat"], "archive_filename": filename,
               "columns": ["TOAs", "betas"], "frames": len(toa), "pages": pages,
               "project_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()}}
    (out / "qin_time_source.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Froze {len(toa)} TOAs and Doppler estimates from {len(pages)} pinned archive pages")


if __name__ == "__main__":
    main()
