"""Transfer the verified first Code 5 word to high-order constellation decisions."""

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "learning")]
from project_paths import experiment_output, require_run_context
from audit_code5_byte_records import BIT_LENGTH, START_RESOURCE
from audit_pn_padding_starts import CacheOnly, demap_exact, lattice_model, word_windows
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_qin_mixed_slot import TYPES
from audit_ut_cross32_labels import labels_as_bits
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


PREFIX = 256


def reference_axis(samples, bits, observed):
    signed = samples[observed] * (1 - 2 * bits[observed].astype(float))
    return complex(np.median(signed.real), np.median(signed.imag))


def shared_counts(left, right, first, second):
    observed = first & second
    return observed.sum(axis=-1), ((left != right) & observed).sum(axis=-1)


def four_corner_slots(samples, axis):
    full = samples / axis
    outer = np.isclose(abs(full.real), 1 / np.sqrt(2), atol=2e-6, rtol=0)
    outer &= np.isclose(abs(full.imag), 1 / np.sqrt(2), atol=2e-6, rtol=0)
    return outer.reshape(-1, len(NONPILOT)).all(axis=1)


def visible_lanes(valid, start, magnitude, corner):
    if corner is None or not magnitude.any():
        return np.broadcast_to(valid[:, None], (len(valid), len(magnitude)))
    # A four-corner slot supplies sign decisions only.
    slots = np.arange(start, start + len(valid)) // len(NONPILOT)
    return valid[:, None] & ~(corner[slots, None] & magnitude[None, :])


def decisions(samples, axis, allocation, models, edge, corner=None):
    start = allocation["start"]
    width = allocation["width_resources"]
    count = allocation["count"]
    selected = samples[start:start + width * count]
    model, magnitude = models[TYPES[allocation["label"]]]
    normalized = selected / axis
    bits, valid = demap_exact(normalized, model)
    per_symbol = bits.shape[-1]
    seen = visible_lanes(valid, start, magnitude, corner)
    return bits.reshape(count, width * per_symbol), seen.reshape(count, width * per_symbol), valid, edge


def audit_allocation(samples, axis, allocation, models, stop, nominal, wrong, corners):
    old, old_seen, baseline, _ = decisions(samples, 1, allocation, models, stop)
    new, new_seen, transferred, _ = decisions(samples, axis, allocation, models, stop)
    original = word_windows(old, old_seen, nominal[:old.shape[1]], stop)
    verified = word_windows(new, new_seen, nominal[:new.shape[1]], stop)
    wrong_pn = word_windows(new, new_seen, wrong[:new.shape[1]], stop)
    magnitude = models[TYPES[allocation["label"]]][1]
    masked_before = visible_lanes(baseline, allocation["start"], magnitude, corners[0]).reshape(old.shape)
    masked_after = visible_lanes(transferred, allocation["start"], magnitude, corners[1]).reshape(new.shape)
    masked_original = word_windows(old, masked_before, nominal[:old.shape[1]], stop)
    masked_verified = word_windows(new, masked_after, nominal[:new.shape[1]], stop)
    masked_wrong = word_windows(new, masked_after, wrong[:new.shape[1]], stop)
    return {"label": allocation["label"], "start": allocation["start"], "total_symbols": len(baseline),
            "baseline_points": int(baseline.sum()), "transferred_points": int(transferred.sum()),
            "added_points": int((~baseline & transferred).sum()), "lost_points": int((baseline & ~transferred).sum()),
            "shared_bit_differences": int(np.count_nonzero((old != new) & old_seen & new_seen)),
            "baseline_stop_words": np.flatnonzero(original["passed"]).tolist(),
            "transfer_stop_words": np.flatnonzero(verified["passed"]).tolist(),
            "transfer_wrong_pn_words": np.flatnonzero(wrong_pn["passed"]).tolist(),
            "masked_baseline_stop_words": np.flatnonzero(masked_original["passed"]).tolist(),
            "masked_transfer_stop_words": np.flatnonzero(masked_verified["passed"]).tolist(),
            "masked_transfer_wrong_pn_words": np.flatnonzero(masked_wrong["passed"]).tolist()}


def pair_metric(first, second):
    a, a_seen, _, edge = first
    b, b_seen, _, _ = second
    regions = [slice(0, PREFIX), slice(PREFIX, edge), slice(edge, None)]
    receipt = {name: shared_counts(a[:, region], b[:, region], a_seen[:, region], b_seen[:, region])
               for name, region in zip(("prefix", "systematic", "posterior"), regions, strict=True)}
    covered, errors = receipt["prefix"]
    registered = (covered >= .90 * PREFIX) & (errors <= .02 * covered)
    c, e = receipt["posterior"]
    repeated = registered & (c >= .90 * (a.shape[1] - edge)) & (e <= .02 * c)
    return receipt, registered, repeated


def pair_result(pa, pb, layout, models, stops):
    a, axis, ac = pa
    b, bx, bc = pb
    allocations = []
    for alloc in layout:
        stop = stops[str(alloc["label"])]
        baseline = pair_metric(decisions(a, 1, alloc, models, stop, ac[0]),
                               decisions(b, 1, alloc, models, stop, bc[0]))
        outcome = pair_metric(decisions(a, axis, alloc, models, stop, ac[1]),
                              decisions(b, bx, alloc, models, stop, bc[1]))
        receipt, registered, repeated = outcome
        added = repeated & ~baseline[2]
        rows = []
        for w in np.flatnonzero(repeated):
            counts = {name: [int(known[w]), int(errors[w])] for name, (known, errors) in receipt.items()}
            rows.append({"word": int(w), "resource": alloc["start"] + int(w) * alloc["width_resources"],
                         "new_registration_or_match": bool(added[w]), **counts})
        allocations.append({"label": alloc["label"], "start": alloc["start"], "words": alloc["count"],
                            "baseline_registered": int(baseline[1].sum()),
                            "transfer_registered": int(registered.sum()), "baseline_repeated": int(baseline[2].sum()),
                            "transfer_repeated": int(repeated.sum()), "new_repeated": int(added.sum()),
                            "lost_repeated": int((baseline[2] & ~repeated).sum()), "repeated_words": rows})
    return allocations


def summarize(rows):
    totals = defaultdict(Counter)
    for row in rows:
        part = "calibration" if row["calibration"] else "held"
        for alloc in row["allocations"]:
            totals[part].update({key: alloc[key] for key in ("total_symbols", "baseline_points", "transferred_points",
                                                            "added_points", "lost_points", "shared_bit_differences")})
            before = set(alloc["baseline_stop_words"])
            after = set(alloc["transfer_stop_words"])
            totals[part].update({"new_stop_words": len(after - before), "lost_stop_words": len(before - after),
                                 "baseline_stop_words": len(before), "transfer_stop_words": len(after),
                                 "wrong_pn_words": len(alloc["transfer_wrong_pn_words"])})
            before_masked = set(alloc["masked_baseline_stop_words"])
            after_masked = set(alloc["masked_transfer_stop_words"])
            totals[part].update({"masked_baseline_stops": len(before_masked), "masked_transfer_stops": len(after_masked),
                                 "masked_new_stops": len(after_masked - before_masked),
                                 "masked_lost_stops": len(before_masked - after_masked),
                                 "masked_wrong_pn": len(alloc["masked_transfer_wrong_pn_words"])})
    return {key: dict(values) for key, values in totals.items()}


def main():
    require_run_context()
    inputs = {}

    def load(run, exp, name):
        return read(checked_artifact(run, exp, name, inputs))

    pad = load("v068-pn-padding-starts", "E070", "pn_padding_starts.json")
    repeats = load("v072-padding-parity-final", "E074", "padding_parity_transfer.json")
    frame_clock = load("v078-archive-code5-links", "E080", "archive_code5_recovery.json")
    cross = load("v028-ut-cross32-labels-final", "E030", "ut_cross32_labels.json")
    with np.load(checked_artifact("v026-ut-qam-extract-repeat", "E028", "ut_qam_symbols.npz", inputs),
                 allow_pickle=False) as data:
        p16, m16 = data["constellation_points"], data["frozen_gray_mapping"]
    amplitude = np.ptp(m16[np.isclose(abs(p16), abs(p16).max())], axis=0) == 0
    points32 = np.array([complex(p["i"], p["q"]) for p in cross["cross32_mapping"]])
    labels32 = labels_as_bits(np.array([p["label"] for p in cross["cross32_mapping"]]), 5)
    models = {"16QAM": (lattice_model(p16, m16), amplitude),
              "cross32": (lattice_model(points32, labels32), np.zeros(5, dtype=bool))}
    observations = {}
    for run, exp, name in (("v076-code5-byte-words", "E078", "archived_code5_words.npz"),
                           ("v078-archive-code5-links", "E080", "recovered_code5_words.npz")):
        with np.load(checked_artifact(run, exp, name, inputs), allow_pickle=False) as z:
            for key in z.files:
                if key.endswith("_word_0_bits"):
                    index = int(key.split("_")[1])
                    observations[index] = np.unpackbits(z[key]), np.unpackbits(z[key[:-4] + "observed"]).astype(bool)
    expected = cross["frozen_pn_model"]
    coords = np.arange(16385)
    pn = pn_bits()
    nominal = equality_predictions(coords, pn, expected)
    wrong = equality_predictions(coords, pn, {**expected, "phase": expected["phase"] + 1})
    lists = [frame["allocations"] for frame in pad["word_windows_and_onsets"]]
    stops = pad["marker_systematic_ends"]
    signs = 1 - 2 * encode_block(np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)])[:48].astype(float)
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    axis_word = slice(START_RESOURCE, START_RESOURCE + BIT_LENGTH)
    reader = CacheOnly(ROOT / ".cache/qin-mat-pages", False)
    rows, pairs = [], []
    with reader, h5py.File(reader, "r") as archive:
        source = archive["yDataDec"]
        if source.shape != (1009, 301, 1024) or set(observations) != set(range(len(source))):
            raise ValueError("Expected a qualified Code 5 word for each source frame")

        def frame_data(i, grid):
            flat = remove_header_gain(grid * rotation, signs)
            axis = reference_axis(flat[axis_word], *observations[i])
            if not np.isfinite(axis) or axis.real <= 0:
                raise ValueError(f"Invalid Code 5 reference in frame {i}")
            return flat, axis, (four_corner_slots(flat, 1), four_corner_slots(flat, axis))

        for start in range(0, len(source), source.chunks[0]):
            stop = min(start + source.chunks[0], len(source))
            packed = as_complex(source[start:stop, 1:, :])[..., NONPILOT % 1024]
            for i, grid in enumerate(packed, start):
                flat, axis, corners = frame_data(i, grid)
                allocs = [audit_allocation(flat, axis, row, models, stops[str(row["label"])], nominal, wrong, corners)
                          for row in lists[i]]
                clock = frame_clock["frames"][i]
                rows.append({"frame": i, "group": clock["group"], "calibration": clock["calibration"],
                             "axis": [axis.real, axis.imag], "allocations": allocs})
            if start % 130 == 0:
                print("Frames", stop, "of 1009", flush=True)

        @lru_cache(maxsize=4)
        def selected(i):
            return frame_data(i, as_complex(source[i, 1:, :])[..., NONPILOT % 1024])

        clock_pairs = {(row["source"], row["destination"]) for row in frame_clock["linked_codewords"]}
        all_pairs = {tuple(row["frames"]) for row in repeats["nearby_diagnostic"]["same_header_pairs"]} | clock_pairs
        for a, b in sorted(all_pairs):
            row = {"frames": [a, b], "e080_clock_link": (a, b) in clock_pairs,
                   "same_first_code5": bool(np.array_equal(observations[a][0], observations[b][0])),
                   "groups": sorted({rows[a]["group"], rows[b]["group"]}),
                   "allocations": pair_result(selected(a), selected(b), lists[a], models, stops)}
            pairs.append(row)
    prior = {r["bytes"][0]: r["sha256"] for r in pad["source"]["pages"]}
    if any(prior[offset] != row["sha256"] for offset, row in reader.receipts.items() if offset in prior):
        raise ValueError("Archive bytes differ from E070")
    report = {"schema": "starlink-code5-gain-transfer/v1", "summary": summarize(rows), "frames": rows, "pairs": pairs,
              "source": {"url": URL, "etag": ETAG, "bytes": SIZE,
                         "pages": [reader.receipts[i] for i in sorted(reader.receipts)]},
              "inputs_sha256": {p.relative_to(ROOT).as_posix(): digest for p, digest in inputs.items()}}
    out = experiment_output("E081")
    out.mkdir(parents=True, exist_ok=True)
    (out / "code5_gain_transfer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
