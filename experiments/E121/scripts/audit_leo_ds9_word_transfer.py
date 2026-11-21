"""Check frozen wideband tail words against both archived LEO DS9 visits."""

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_leo_ds7_reference_transfer import PERIOD, rotation_class, sign_invariant
from audit_ut_repeated_data import checked_artifact, read


SOURCE = Path("reports/2026_09_28_sequence_semantics")
VISITS = {"middle": "ds9_word_audit.json", "last": "ds9_last_word_audit.json"}
WINDOW_STARTS = range(14, 271, 32)
SHUFFLE_SEED = 20261121121
SHUFFLES = 999


def rotate(word, shift):
    return word[shift:] + word[:shift]


def codebook(seed):
    return {
        "".join("1" if a == b else "0" for a, b in zip(seed, rotate(seed, k), strict=True)): k
        for k in range(PERIOD)
    }


def read_source(path, digests):
    relative = SOURCE / path
    payload = (EXTERNAL_LEO_ROOT / relative).read_bytes()
    digests[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
    return payload


def audit_visit(receipt, book, native_book):
    frames = receipt["frames"]
    expected = {(frame, first) for frame in frames for first in WINDOW_STARTS}
    seen = set()
    frame_words = {frame: set() for frame in frames}
    rows = []
    for row in receipt["rows"]:
        frame, first = row["frame"], row["first_symbol"]
        coordinate = (frame, first)
        word, peer = row["word"], row["peer_word"]
        gate = row["full"] and word == peer and min(row["score0"], row["held"]) > 0.25
        if (
            coordinate not in expected or coordinate in seen
            or row["last_symbol"] != min(first + 31, 300)
            or any(len(x) != PERIOD or set(x) - {"0", "1"} for x in (word, peer))
            or gate != row["accepted"]
        ):
            raise ValueError(f"Inconsistent DS9 window {coordinate}")
        seen.add(coordinate)
        phase = book.get(word)
        if gate:
            frame_words[frame].add(word)
        rows.append({
            "frame": frame, "first_symbol": first, "last_symbol": row["last_symbol"],
            "word": word, "peer_word": peer, "full": row["full"],
            "score0": row["score0"], "held": row["held"], "accepted": gate,
            "original_phase": row["phase"], "predicted_phase": phase,
            "native_origin_match": word in native_book,
        })
    if seen != expected or len(frames) != len(frame_words):
        raise ValueError("Missing or duplicate DS9 frame/windows")
    passed = [r for r in rows if r["accepted"]]
    return {
        "cached_soft_sha256_expected": receipt["input_sha256"],
        "windows": len(rows),
        "accepted_windows": len(passed),
        "matching_windows": sum(r["predicted_phase"] is not None for r in passed),
        "exact_reported_phase_windows": sum(
            r["predicted_phase"] is not None and r["predicted_phase"] == r["original_phase"]
            for r in passed
        ),
        "native_origin_matches": sum(r["native_origin_match"] for r in passed),
        "rejected_words_in_book": sum(r["predicted_phase"] is not None for r in rows if not r["accepted"]),
        "frames": len(frames),
        "matching_frames": sum(bool(words) and words.issubset(book) for words in frame_words.values()),
        "conflicting_frames": [frame for frame, words in frame_words.items() if len(words) > 1],
        "unique_words": len({r["word"] for r in passed}),
        "rows": rows,
    }


def main():
    require_run_context()
    inputs, external = {}, {}
    tail = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    template = tail["cycle_folds"][0]["signed_template"]
    if len(template) != PERIOD or set(template) != {-1, 1}:
        raise ValueError("Expected a complete signed wideband template")
    seed = "".join("1" if value > 0 else "0" for value in template)
    tree = ast.parse(read_source("firmware_seed_search.py", external))
    origin = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SEED" for t in node.targets)
    )
    shifts = [i for i in range(PERIOD) if rotate(seed, i) == origin]
    if len(shifts) != 1:
        raise ValueError("Wideband and LEO seeds do not define a unique origin")
    book, native_book = codebook(rotate(seed, shifts[0])), codebook(seed)
    if len(book) != PERIOD:
        raise ValueError("The wideband dictionary does not have 60 unique states")
    visits = {
        name: audit_visit(json.loads(read_source(f"local/{filename}", external)), book, native_book)
        for name, filename in VISITS.items()
    }
    observed = {
        sign_invariant(rotation_class(row["word"]))
        for visit in visits.values() for row in visit["rows"] if row["accepted"]
    }
    rng = random.Random(SHUFFLE_SEED)
    controls = Counter()
    for _ in range(SHUFFLES):
        words = codebook(rng.sample(seed, PERIOD))
        control = {sign_invariant(rotation_class(word)) for word in words}
        controls[len(observed & control)] += 1
    result = {
        "schema": "starlink-leo-ds9-word-transfer/v1",
        "wideband_inputs_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): d for p, d in inputs.items()},
        "leo_inputs_sha256": external,
        "leo_word_report_base": SOURCE.as_posix(),
        "origin_left_shift": shifts[0],
        "wideband_seed": seed,
        "leo_seed": origin,
        "distinct_rotation_classes_modulo_sign": len(observed),
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_class_hit_histogram": dict(sorted(controls.items())),
        "visits": visits,
    }
    output = experiment_output("E121")
    output.mkdir(parents=True, exist_ok=True)
    (output / "leo_ds9_word_transfer.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, visit in visits.items():
        print(f"{name}: {visit['matching_frames']}/{visit['frames']} frames, "
              f"{visit['exact_reported_phase_windows']}/{visit['accepted_windows']} accepted window phases, "
              f"native-origin matches {visit['native_origin_matches']}")
    print(f"Observed rotation/sign classes {len(observed)}, shuffled maximum {max(controls)}")


if __name__ == "__main__":
    main()
