"""Ideal erasure of published H2 codewords through two narrow receiver bands."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import LEO_SNAPSHOT, RAW_WAV, experiment_output, require_run_context
from audit_code_observability import encoder_matrix
from code68_mapping_core import nullspace
from educational_fec_demo import crc8
from layout_core import NONPILOT, parse_descriptors, resource_geometry, second_header_start

GRID = 234375
LEGACY = "runs/v008-complete-header-layout-02/artifacts/experiments/E010/results/full_header_layout.json"
UT = "runs/v022-ut-replay/artifacts/experiments/E014/results/ut_transfer.json"
UT_PATH = "data/raw/ut_250msps/Starlink_Downlink_IQ_250MSps.bin"
TRACK = "experiments/E001/results/refined_track.npz"
WAV_MODEL = "experiments/E006/results/narrowband_sync_prefix_extended_test.json"
SELECTION_ROOT = LEO_SNAPSHOT / "reports/2026_10_18_dsx01_pilot_clock"


def header_bits(words):
    return np.array([int(bit) for word in words for bit in f"{int(word, 16):032b}"], dtype=np.uint8)


def header_hex(bits):
    words = bits.reshape(-1, 32)
    return [f"0x{int(''.join(map(str, word)), 2):08X}" for word in words]


def attach_crc(bits):
    value = crc8(bits[:-8])
    bits[-8:] = [(value >> b) & 1 for b in range(7, -1, -1)]


def visible_positions(start, count, carriers):
    _, _, k = resource_geometry(start, count)
    return tuple(np.flatnonzero(np.isin(k, carriers)).tolist())


@lru_cache(maxsize=None)
def encoding(blocks):
    return encoder_matrix(blocks)


@lru_cache(maxsize=None)
def affine_directions(blocks, adjustable):
    """CRC differences are linear even though the initialized CRC is affine."""
    width = 32 * blocks
    origin = np.zeros(width, dtype=np.uint8)
    attach_crc(origin)
    deltas = np.repeat(origin[None, :], len(adjustable), axis=0)
    for row, pos in zip(deltas, adjustable, strict=True):
        row[pos] ^= 1
        attach_crc(row)
    deltas ^= origin
    return deltas, (encoding(blocks) @ deltas.T) % 2


def affine_projection(reference, adjustable, observed):
    if reference.ndim != 1 or not len(reference) or len(reference) % 32:
        raise ValueError("Unaligned header")
    blocks = len(reference) // 32
    expected = reference.copy()
    attach_crc(expected)
    if not np.array_equal(expected, reference):
        raise ValueError("Reference CRC mismatch")
    deltas, projected = affine_directions(blocks, adjustable)
    equations = [
        int.from_bytes(row, "little")
        for row in np.packbits(projected[list(observed)], axis=1, bitorder="little")
    ]
    kernel, rank = nullspace(equations, len(adjustable))
    union = 0
    for vector in kernel:
        union |= vector
    witness = None
    for vector in kernel:
        chosen = [row for row in range(len(adjustable)) if (vector >> row) & 1]
        trial = reference ^ np.bitwise_xor.reduce(deltas[chosen], axis=0)
        layout = parse_descriptors(trial, 20)
        if layout["packing_valid"] and all(item["quantity"] > 0 for item in layout["entries"]):
            assert not np.any((encoding(blocks)[list(observed)] @ (trial ^ reference)) % 2)
            expected = trial.copy()
            attach_crc(expected)
            assert np.array_equal(expected, trial)
            witness = header_hex(trial)
            break
    return {
        "adjustable_bits": len(adjustable),
        "observed_coordinates": len(observed),
        "observable_rank": rank,
        "free_dimensions": len(kernel),
        "individually_identified_positions": [
            bit for index, bit in enumerate(adjustable) if not (union >> index) & 1
        ],
        "synthetic_crc_compatible_alternative_hex": witness,
    }


def band_masks():
    wav_model = read_json(ROOT / WAV_MODEL)
    with np.load(ROOT / TRACK, allow_pickle=False) as track:
        t = track["time_s"][track["fit_mask"].astype(bool)]
        cfo = np.polyval(track["quadratic_coefficients"], t)
    offset = wav_model["exact"]["carrier_offset_from_moving_line_hz"]
    wav_passband = abs(NONPILOT[None, :] * GRID + offset + cfo[:, None]) <= 1200000
    wav_carriers = tuple(NONPILOT[wav_passband.any(axis=0)].tolist())
    masks = {"wav": wav_carriers}
    metadata = {"wav_track_interval_s": [float(t.min()), float(t.max())], "wav_model_offset_hz": offset}
    edge_union = set()
    for dataset in ("DSX01-S05", "paired_RX"):
        selection = read_json(SELECTION_ROOT / dataset / "selection.json")
        for item in selection["windows"]:
            offset = item["winner"]["fractional_tracking_cfo_hz"]
            frequencies = (NONPILOT + 1024 - 531.5) * GRID + offset
            carriers = tuple(NONPILOT[abs(frequencies) <= selection["sample_rate_hz"] / 2].tolist())
            masks[f"{dataset}/iq_{item['number']}"] = carriers
            edge_union.update(carriers)
    masks["npz_union"] = tuple(sorted(edge_union))
    masks["optimistic_wav_npz_union"] = tuple(sorted(edge_union.union(wav_carriers)))
    return masks, metadata


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_records(report, capture):
    if capture == "legacy480":
        for frame in report["frames"]:
            if frame["second_header"]["crc_ok"]:
                yield frame["legacy_frame_index"], frame["first_header"], frame["second_header"]
    else:
        for frame in report["candidates"]:
            if frame.get("h2", {}).get("crc_ok"):
                yield frame["candidate_index"], frame["h1"], {
                    **frame["h2"], "start_resource": frame["predicted_h2_start"]
                }


def project_capture(report, capture, masks):
    rows = []
    for frame, h1, h2 in capture_records(report, capture):
        start = h2["start_resource"]
        if second_header_start(header_bits(h1["words_hex"])) != start:
            raise ValueError("H2 placement disagrees with H1")
        words = h2["words_hex"]
        bits = header_bits(words)
        blocks = len(words)
        layout = parse_descriptors(bits, 20)
        if not layout["packing_valid"] or not layout["entries"]:
            raise ValueError("Reference does not satisfy working layout")
        stop = 16 + 20 * len(layout["entries"])
        adjustable = tuple(range(16, stop))
        quantities = tuple(i for i in adjustable if (i - 16) % 20 >= 8)
        encoded = encoding(blocks) @ bits % 2
        projections = {}
        for label, carriers in masks.items():
            observed = visible_positions(start, len(encoded), carriers)
            projections[label] = {
                "positions": observed,
                "projected_reference": "".join(map(str, encoded[list(observed)])),
                "known_prefix_and_padding": affine_projection(bits, adjustable, observed),
                "also_known_mcs": affine_projection(bits, quantities, observed),
            }
        full = affine_projection(bits, adjustable, tuple(range(len(encoded))))
        rows.append({
            "capture": capture, "frame": frame, "start_resource": start,
            "prefix_hex": words[0][:6], "blocks": blocks, "entries": layout["entries"],
            "reference_words_hex": words, "masks": projections,
            "full_band_free_dimensions": full["free_dimensions"],
        })
    return rows


def same_geometry_pairs(rows, masks):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["capture"], row["start_resource"], row["blocks"], row["prefix_hex"])
        groups[key].setdefault(tuple(row["reference_words_hex"]), row)
    pairs = [pair for group in groups.values() for pair in combinations(group.values(), 2)]
    return {
        "tested_deduplicated_pairs": len(pairs),
        "pairs_by_mask": {
            mask: [
                {"capture": a["capture"], "frames": [a["frame"], b["frame"]],
                 "visible_coordinates": len(a["masks"][mask]["positions"])}
                for a, b in pairs
                if a["masks"][mask]["projected_reference"] == b["masks"][mask]["projected_reference"]
            ]
            for mask in masks
        },
    }


def finite_geometry(masks):
    """Enumerate every second-header start under the frozen first-body model."""
    cases = []
    capacity = 300 * len(NONPILOT)
    for count in range(1, (capacity - 228) // 2048 + 1):
        start = 228 + 2048 * count
        for blocks in range(2, 8):
            if start + 114 * blocks <= capacity:
                cases.append((start, blocks))
    summaries = {}
    for name, carriers in masks.items():
        coverage = [len(visible_positions(start, 114 * blocks, carriers)) for start, blocks in cases]
        summaries[name] = {
            "maximum_visible_coordinates": max(coverage),
            "cases_with_no_visible_coordinates": coverage.count(0),
            "cases_with_visible_coordinates": len(cases) - coverage.count(0),
        }
    return {
        "h2_positions_and_lengths": len(cases),
        "block_range": [2, 7],
        "masks": summaries,
    }


def main():
    require_run_context()
    masks, band_metadata = band_masks()
    input_paths = [ROOT / x for x in (LEGACY, UT, TRACK, WAV_MODEL, UT_PATH, "provenance/ut_capture_import.json")]
    input_paths.extend([RAW_WAV, *(SELECTION_ROOT / x / "selection.json" for x in ("DSX01-S05", "paired_RX"))])
    inputs = {p.relative_to(ROOT).as_posix(): sha256(p) for p in input_paths}
    ut_report = read_json(ROOT / UT)
    if inputs[UT_PATH] != ut_report["source_sha256"]:
        raise ValueError("250 MS/s reference has a different parent recording")
    rows = project_capture(read_json(ROOT / LEGACY), "legacy480", masks)
    rows.extend(project_capture(ut_report, "ut250", masks))
    result = {
        "schema": "starlink-h2-projection/v1", "inputs_sha256": inputs,
        "band_metadata": band_metadata, "carrier_masks": masks, "frames": rows,
        "finite_geometry": finite_geometry(masks), "same_geometry_collisions": same_geometry_pairs(rows, masks),
        "assumptions": [
            "Ideal band-centre visibility, orthogonal resource grid and perfect in-band hard bit/phase.",
            "H1 count and H2 placement, block count, complete prefix and zero padding supplied by reference.",
            "Band unions are oracle views of the same H2 bits, not actual simultaneous measurements.",
            "Synthetic alternatives obey CRC, working packing and positive quantity, without downstream constraints.",
            "No original narrowband bits estimated; measured wideband reference words are reused.",
        ],
    }
    out = experiment_output("E024")
    out.mkdir(parents=True, exist_ok=True)
    (out / "h2_projection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Projected {sum(r['capture'] == 'legacy480' for r in rows)} legacy and "
          f"{sum(r['capture'] == 'ut250' for r in rows)} UT headers.", flush=True)


if __name__ == "__main__":
    main()
