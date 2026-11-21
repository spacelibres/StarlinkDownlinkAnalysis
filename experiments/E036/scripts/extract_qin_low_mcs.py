"""Extract low-MCS candidates and measure MCS 93 from all available target frames."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_qin_exemplars import remove_header_gain
from audit_qin_lengths import detect_terminal, header_geometry, regions_for
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_reference
from snapshot_qin_exemplars import ArchiveReader, ETAG, SIZE, URL, as_complex

TARGET = (6, 35, 42, 93)


def target_symbols(dataset, target_indices, p, signs):
    by_group = defaultdict(list)
    step = dataset.chunks[0]
    for frame in target_indices:
        by_group[frame//step].append(frame)
    for group, indices in by_group.items():
        data = as_complex(dataset[indices, 1:, :])
        for frame, values in zip(indices, data, strict=True):
            yield frame, group*step, remove_header_gain(values[:, NONPILOT % 1024] * p, signs)


def main():
    require_run_context()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--allow-unverified-source", action="store_true")
    args = cli.parse_args()
    inputs = {}
    survey = read(checked_artifact("v034-qin-header-survey", "E036", "qin_header_survey.json", inputs))
    widths = read(checked_artifact("v033-qin-lengths-final", "E035", "qin_lengths.json", inputs))
    precedence = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    known_widths = {int(k): v for k, v in {**widths["inherited_units"], **widths["fitted_units"]}.items()}
    heads = {v["frame"]: v for v in survey["frames"] if v["h2"].get("crc_ok") is True}
    eligible = [i for i, head in heads.items() if set(header_geometry(head)[1]).intersection(TARGET)]
    p = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2*encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    template = np.array(precedence["cycle_folds"][0]["signed_template"], dtype=float)
    cache = ROOT / ".cache/qin-mat-pages"
    cache.mkdir(parents=True, exist_ok=True)
    reader = ArchiveReader(cache, args.allow_unverified_source)
    observed = {}
    rows = []
    with reader, h5py.File(reader, "r") as stream:
        for frame, group, flat in target_symbols(stream["yDataDec"], eligible, p, signs):
            terminal = detect_terminal(flat, template)
            measured = terminal["phase_refined_edge"]
            first, amounts = header_geometry(heads[frame])
            partial = first + sum(known_widths.get(code, 0)*q for code, q in amounts.items())
            rows.append({"frame": frame, "group": group, "measured_edge": measured,
                         "fit_without_93": partial, "count_93": amounts.get(93, 0), "terminal": terminal})
            observed[frame] = flat.astype(np.complex64)
    fit = next((row for row in rows if row["count_93"] and row["measured_edge"] is not None), None)
    if fit is None:
        raise ValueError("No validated MCS 93 endpoint")
    unit, remainder = divmod(fit["measured_edge"] - fit["fit_without_93"], fit["count_93"])
    if remainder or unit <= 0:
        raise ValueError("No positive integer MCS 93 unit")
    units = {**known_widths, 93: unit}
    results = {code: defaultdict(list) for code in TARGET}
    for row in rows:
        frame = row["frame"]
        regions = regions_for(heads[frame], units)
        edge = regions[-1][-1]
        row["model_end"] = edge
        row["endpoint_supported"] = (edge == row["measured_edge"])
        if not row["endpoint_supported"]:
            continue
        for code, count, start, stop in regions:
            if code not in results:
                continue
            block = observed[frame][start:stop].reshape(count, units[code])
            store = results[code]
            store["symbols"].extend(block)
            store["frames"].extend([frame] * count)
            store["groups"].extend([row["group"]] * count)
            store["resource_start"].extend(start + np.arange(count) * units[code])
            store["word_in_allocation"].extend(range(count))
            store["is_final"].extend(j == count - 1 for j in range(count))
    out = experiment_output("E036")
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"mcs{code}_{name}": np.asarray(data)
              for code, contents in results.items() for name, data in contents.items()}
    report = {"schema": "starlink-qin-low-mcs-extract/v1", "boundary_rows": rows,
              "input_hashes": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()},
              "resource_widths": {c: units[c] for c in TARGET}, "mcs93_fit_frame": fit["frame"],
              "mcs93_fit_group": fit["group"], "retained_words": {c: len(results[c]["frames"]) for c in TARGET},
              "archive": {"url": URL, "etag": ETAG, "bytes": SIZE,
                          "local_range_cache_transport": "The original range cache used unverified TLS.",
                          "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
              "scope": "complex64 frequency-domain symbols after frozen pilot PN and H1 normalization, "
                       "only from frames with full endpoint agreement."}
    (out / "qin_low_mcs_extract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(out / "qin_low_mcs_symbols.npz", **arrays)
    print("MCS 93 unit", unit, "from frame", fit["frame"], "; retained words", report["retained_words"], flush=True)


if __name__ == "__main__":
    main()
