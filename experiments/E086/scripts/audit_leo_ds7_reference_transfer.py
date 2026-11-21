"""Test a frozen wideband tail model on the LEO DS7 bit exports."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read


DS7 = Path("reports/2026_09_27_ds7_header/local")
PERIOD = 60
SEED = 2026112188
CONTROL_COUNT = 999


def rotation_class(word):
    return min(word[phase:] + word[:phase] for phase in range(PERIOD))


def complement(word):
    return word.translate(str.maketrans("01", "10"))


def classes(template):
    """Enumerate phases using the positive-one convention for -S[r] S[r-d]."""
    phases = defaultdict(list)
    for phase in range(PERIOD):
        shifted = template[phase:] + template[:phase]
        word = "".join("1" if a != b else "0" for a, b in zip(template, shifted, strict=True))
        phases[rotation_class(word)].append(phase)
    return dict(phases)


def sign_invariant(key):
    return min(key, rotation_class(complement(key)))


def main():
    require_run_context()
    inputs = {}
    tail = read(checked_artifact("v023-binary-tail-final", "E025", "ut_binary_tail.json", inputs))
    families = read(checked_artifact(
        "v047-reference-word-families", "E049", "reference_word_families.json", inputs
    ))
    template = tail["cycle_folds"][0]["signed_template"]
    if len(template) != PERIOD or set(template) != {-1, 1}:
        raise ValueError("Expected a complete signed wideband reference")
    predicted = classes(template)

    ext_inputs = {}

    def source(relative):
        data = (EXTERNAL_LEO_ROOT / DS7 / relative).read_bytes()
        ext_inputs[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    hypothesis_name = "even-parity-hypothesis.json"
    hypothesis = source(hypothesis_name)
    validation = source("even-parity-validation.json")
    if validation["hypothesis_sha256"] != ext_inputs[hypothesis_name]:
        raise ValueError("LEO hypothesis differs from its validation receipt")
    digests = {"/".join(k.split("/")[-2:]): v for k, v in validation["input_sha256"].items()}
    frames_held = set(hypothesis["validation_frames"])
    group_held = hypothesis["validation_group"]
    captures, rows = {}, []
    for group, discovery in hypothesis["discovery_frames"].items():
        name = f"{group}/tcodes.json"
        exports = source(name)["results"]
        inventory = source(f"{group}/inventory.json")
        if digests[name] != ext_inputs[name]:
            raise ValueError(f"LEO export differs from its validation receipt: {name}")
        capture = []
        for item in inventory["exports"]:
            capture.append({
                "session": item["session_id"],
                "visit": item["visit"]["event"]["visit_index"],
                "excerpt_sha256_from_inventory": item["excerpt_sha256"],
                "relative_excerpt": f"{group}/{item['name']}.npy",
                "excerpt_locally_available": (EXTERNAL_LEO_ROOT / DS7 / group / f"{item['name']}.npy").is_file(),
            })
        captures[group] = capture
        expected = set(discovery)
        if group == group_held:
            expected.update(frames_held)
        if sorted(row["frame"] for row in exports) != sorted(expected):
            raise ValueError(f"Missing, duplicate, or unexpected DS7 frame in {group}")
        for row in exports:
            word = row["combined_word"]
            if (
                len(word) != PERIOD or set(word) - {"0", "1"} or not row["repeated_code_candidate"]
                or row["receiver_bit_agreement"] != 1 or row["rx0_word"] != word or row["rx1_word"] != word
                or row["rx1_heldout_correlation"] <= row["wrong_code_max"]
            ):
                raise ValueError(f"DS7 export failed its recorded gates: {group}/{row['frame']}")
            key = rotation_class(word)
            flipped = rotation_class(complement(word))
            rows.append({
                "group": group, "frame": row["frame"], "rotation_class": key,
                "partition": "prospective" if group == group_held and row["frame"] in frames_held else "discovery",
                "matching_phases": predicted.get(key, predicted.get(flipped, [])),
                "expected_polarity_exact": key in predicted,
                "opposite_polarity_exact": flipped in predicted,
                "even_parity": word.count("1") % 2 == 0,
            })
    archived = families["wideband"]["qin_families"]
    archive_keys = {f"{int(key, 16) >> 4:060b}" for key in archived}
    for row in rows:
        row["in_e049_archive"] = sign_invariant(row["rotation_class"]) in archive_keys
    observed = {sign_invariant(row["rotation_class"]) for row in rows}
    by_group = {
        group: {
            mode: sum(row[mode] for row in rows if row["group"] == group)
            for mode in ("expected_polarity_exact", "opposite_polarity_exact")
        }
        for group in captures
    }
    rng = random.Random(SEED)
    random_hits = Counter()
    for _ in range(CONTROL_COUNT):
        control = {sign_invariant(key) for key in classes(rng.sample(template, PERIOD))}
        random_hits[len(observed.intersection(control))] += 1
    report = {
        "schema": "starlink-leo-ds7-reference-transfer/v1",
        "project_inputs_sha256": {key.relative_to(PROJECT_ROOT).as_posix(): value for key, value in inputs.items()},
        "leo_report_base": DS7.as_posix(),
        "leo_inputs_sha256": ext_inputs,
        "captures": captures,
        "predicted_classes": predicted,
        "observed_unique_rotations_modulo_sign": len(observed),
        "e049_classes": len(archive_keys),
        "e049_classes_predicted": len(archive_keys.intersection(map(sign_invariant, predicted))),
        "exact_polarity_matches_by_visit": by_group,
        "prospective_matches": sum(bool(r["matching_phases"]) for r in rows if r["partition"] == "prospective"),
        "matches": sum(bool(r["matching_phases"]) for r in rows),
        "all_predicted_even": all(key.count("1") % 2 == 0 for key in predicted),
        "weight_preserving_shuffle_control_seed": SEED,
        "weight_preserving_shuffle_class_hit_histogram": dict(sorted(random_hits.items())),
        "rows": rows,
    }
    output = experiment_output("E086")
    output.mkdir(parents=True, exist_ok=True)
    (output / "leo_ds7_reference_transfer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DS7 words {report['matches']}/{len(rows)}, unique rotation classes {len(observed)}, "
          f"prospective matches {report['prospective_matches']}/{len(frames_held)}, "
          f"control maximum {max(random_hits)}, exact polarity matches {by_group}")


if __name__ == "__main__":
    main()
