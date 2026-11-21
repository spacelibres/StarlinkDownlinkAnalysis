"""Locate MCS 85 record starts without reading the candidate header tail."""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_gain_transfer import decisions
from audit_mcs85_counter import REVERSE8, header_checks
from audit_mcs85_records import WEIGHT, system_bytes
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_cross32_labels import labels_as_bits, nearest_symbols
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


SYSTEMATIC = 12288
PREFIX = 4
WINDOW = 64
WITNESS = 8
MAX_DISTANCE = 2
CONTROL_SHIFTS = (-4, -1, 1, 4)


def prefix_sites(data, observed, prefix):
    trials = sliding_window_view(data, len(prefix))
    complete = (sliding_window_view(observed, len(prefix)) == 255).all(axis=1)
    near = WEIGHT[trials ^ prefix].sum(axis=1) <= MAX_DISTANCE
    return np.flatnonzero(complete & near)


def trained_boundary(frequency):
    # Bytes 4 through 7 are already known to vary, so they are not a witness.
    for k in range(WITNESS + 8, len(frequency) - 4):
        if np.all(frequency[WITNESS:k] >= .8) and np.all(frequency[k:k + 4] < .7):
            return k
    raise ValueError("The calibration groups do not identify a stable-to-variable boundary")


def calibrate(records, streams):
    samples = []
    for r in records["frame_results"]:
        if r["frame"] not in records["frames_split"]["calibration"]:
            continue
        stream = streams[r["frame"]]
        coords = np.array([x["byte_offset"] for x in r["records"] if x["byte_offset"] + WINDOW <= len(stream)])
        local = stream[coords[:, None] + np.arange(WINDOW)]
        samples.append(np.unpackbits(local, axis=1, bitorder="little").mean(axis=0).reshape(WINDOW, 8))
    probability = np.mean(samples, axis=0)
    values = np.packbits(probability >= .5, axis=1, bitorder="little").ravel()
    reliability = np.maximum(probability, 1 - probability).mean(axis=1)
    return values, trained_boundary(reliability), reliability


def score_stream(data, mask, template, boundary, lengths):
    sites = prefix_sites(data, mask, template[:PREFIX])
    valid = sites[(sites >= 4) & (sites + WINDOW + 4 <= len(data))]
    coords = valid[:, None] + np.arange(WINDOW)
    values = data[coords]
    visible = mask[coords]
    correct = WEIGHT[(values ^ template) & visible]
    exposure = WEIGHT[visible]
    ref = correct[:, WITNESS:boundary]
    ver = exposure[:, WITNESS:boundary]
    same = {}
    for shift in CONTROL_SHIFTS:
        candidate = coords[:, WITNESS:boundary] + shift
        common = visible[:, WITNESS:boundary] & mask[candidate]
        base = WEIGHT[(values[:, WITNESS:boundary] ^ template[WITNESS:boundary]) & common]
        other = WEIGHT[(data[candidate] ^ template[WITNESS:boundary]) & common]
        same[str(shift)] = [int(WEIGHT[common].sum()), int(base.sum()), int(other.sum())]
    known = set(sites)
    checks = Counter()
    rows = []
    for i, pos in enumerate(valid):
        tag = values[i, 4:6].tobytes().hex()
        status = None
        if (visible[i, 4:6] == 255).all() and tag in lengths and pos + lengths[tag] + PREFIX <= len(data):
            status = "hit" if pos + lengths[tag] in known else "missed"
            checks[tag + "_" + status] += 1
        rows.append({"offset": int(pos), "prefix_hex": values[i, :boundary].tobytes().hex(),
                     "prefix_mask_hex": visible[i, :boundary].tobytes().hex(),
                     "witness_errors": int(ref[i].sum()), "witness_bits": int(ver[i].sum()), "next_prefix": status})
    return {"prefix_hits": len(sites), "rows": rows, "exposure": exposure.sum(axis=0).tolist(),
            "errors": correct.sum(axis=0).tolist(), "paired_shift_scores": same, "successors": checks}


def stream_from_bits(bits, masks, pn):
    return (system_bytes(bits, SYSTEMATIC, pn).ravel(),
            np.packbits(masks[:, :SYSTEMATIC], axis=1, bitorder="little").ravel())


def archive_streams(source, gain, controls, labels, models):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    points = np.array([complex(p["i"], p["q"]) for p in labels["cross32_mapping"]])
    points /= max(abs(points))
    maps = labels_as_bits(np.array([p["label"] for p in labels["cross32_mapping"]]), 5)

    @lru_cache(maxsize=1)
    def block(start):
        return as_complex(source[start:min(start + source.chunks[0], len(source)), 1:, :])[..., NONPILOT % 1024]

    for row in controls:
        frame = row["frame"]
        first = frame // source.chunks[0] * source.chunks[0]
        flat = remove_header_gain(block(first)[frame - first] * rotation, signs)
        receiver = gain["frames"][frame]
        if frame != receiver["frame"] or row["group"] != receiver["group"]:
            raise ValueError("Archive gain and reference allocation disagree")
        axis = complex(*receiver["axis"])
        start, width, words = (row[k] for k in ("resource_start", "resources_per_word", "words"))
        view = (flat[start:start + width * words] / axis).reshape(words, width)
        bits = maps[nearest_symbols(view, points)].reshape(words, -1)
        lut, seen, _, _ = decisions(flat, axis, {"start": start, "width_resources": width,
                                               "count": words, "label": 85}, models, SYSTEMATIC)
        if np.any(lut[seen] != bits[seen]):
            raise ValueError("Archive lattice witness changed")
        yield row, bits, seen


def protocol_checks(data, records):
    selected = []
    reversed_data = REVERSE8[data].astype(np.uint8)
    for row in records["length_predictions"]:
        if row["status"] != "matched":
            continue
        a, b = row["reference"], row["predicted"]
        selected.append({"start": a, "stop": b,
                         "lsb": header_checks(data, a, b), "msb": header_checks(reversed_data, a, b)})
    return selected


def projection(rows, windows, boundary):
    carriers = []
    for row in rows:
        for hit in row["result"]["rows"]:
            offsets = hit["offset"] + np.r_[np.arange(PREFIX), np.arange(WITNESS, boundary)]
            words, bits = np.divmod((offsets[:, None] * 8 + np.arange(8)).ravel(), SYSTEMATIC)
            resource = row["resource_start"] + words * row["resources_per_word"] + bits // 5
            carriers.append(NONPILOT[resource % len(NONPILOT)])
    positions = np.asarray(carriers).reshape(-1, (PREFIX + boundary - WITNESS) * 8)
    return [{"kind": w["kind"], "labels": w["labels"],
             "complete": int(np.isin(positions, w["carriers"]).all(axis=1).sum()), "sites": len(positions)}
            for w in windows]


def main():
    require_run_context()
    inputs = {}

    def art(run, exp, name):
        return checked_artifact(run, exp, name, inputs)

    records = read(art("v050-mcs85-spacing", "E052", "mcs85_records.json"))
    labels = read(art("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json"))
    pn_model = records["pn_model"]
    if pn_model != labels["frozen_pn_model"]:
        raise ValueError("PN receipts differ")
    pn = equality_predictions(np.arange(SYSTEMATIC), pn_bits(), pn_model)
    wrong_pn = equality_predictions(np.arange(SYSTEMATIC), pn_bits(), {**pn_model, "phase": pn_model["phase"] + 1})
    run = "v123-cross32-partial-decode"
    decoder = read(art(run, "E125", "cross32_partial_decode.json"))["formats"]["85"]["raw250"]
    frames = np.asarray(decoder["frames"])
    packed = art(run, "E125", "cross32_partial_decisions.npz")
    with np.load(packed, allow_pickle=False) as z:
        raw, visible = (np.unpackbits(z[f"85_decoded_{name}"], axis=1) for name in ("bits", "seen"))
    streams = {int(f): stream_from_bits(raw[frames == f], visible[frames == f], pn) for f in np.unique(frames)}
    template, boundary, reliability = calibrate(records, {f: v[0] for f, v in streams.items()})
    lengths = records["fitted_distances_by_exact_label"]
    raw_reports = []
    for r in records["frame_results"]:
        frame = r["frame"]
        if frame in records["frames_split"]["reserved"]:
            data, mask = streams[frame]
            wrong, _ = stream_from_bits(raw[frames == frame], visible[frames == frame], wrong_pn)
            raw_reports.append({"frame": frame, "result": score_stream(data, mask, template, boundary, lengths),
                                "wrong_PN_prefix_hits": len(prefix_sites(wrong, mask, template[:PREFIX])),
                                "protocol_checks": protocol_checks(data, r)})
    gain = read(art("v079-code5-gain-masked", "E081", "code5_gain_transfer.json"))
    controls = read(art("v122-mcs85-archive-records-verified", "E124", "mcs85_archive_records.json"))
    exclude = set(controls["archive_overlapping_groups_excluded"])
    held = [r for r in controls["rows"] if not r["calibration"] and r["group"] not in exclude]
    models = constellation_models(labels, inputs)
    archive_reports = []
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    with reader, h5py.File(reader, "r") as archive:
        for r, bits, seen in archive_streams(archive["yDataDec"], gain, held, labels, models):
            data, mask = stream_from_bits(bits, seen, pn)
            wrong, _ = stream_from_bits(bits, seen, wrong_pn)
            archive_reports.append({key: r[key] for key in ("frame", "group", "resource_start", "resources_per_word")}
                                   | {"result": score_stream(data, mask, template, boundary, lengths),
                                      "wrong_PN_prefix_hits": len(prefix_sites(wrong, mask, template[:PREFIX]))})
    windows = read(art("v120-cross-mcs-checks", "E122", "cross_mcs_checks.json"))["leo"]
    result = {"schema": "starlink-mcs85-record-boundary/v1", "word_bits": SYSTEMATIC,
              "template_hex": template.tobytes().hex(), "training_modal_agreement": reliability.tolist(),
              "learned_boundary": boundary, "prefix_bytes": PREFIX, "prefix_max_errors": MAX_DISTANCE,
              "raw_split": records["frames_split"], "raw_split_seed": records["inherited_split_seed"],
              "archive_split_seed": controls["archive_group_seed"], "archive_excluded_groups": sorted(exclude),
              "raw": raw_reports, "archive": archive_reports, "leo": projection(archive_reports, windows, boundary),
              "archive_source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                                 "pages": [reader.receipts[k] for k in sorted(reader.receipts)]},
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): d for p, d in inputs.items()}}
    output = experiment_output("E126")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mcs85_record_boundary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Boundary", boundary, "prefix", template[:PREFIX].tobytes().hex(), flush=True)
    for key in ("raw", "archive"):
        print(key, sum(row["result"]["prefix_hits"] for row in result[key]), "prefix hits", flush=True)


if __name__ == "__main__":
    main()
