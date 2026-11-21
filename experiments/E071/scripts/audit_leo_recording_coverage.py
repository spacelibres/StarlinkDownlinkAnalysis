"""Audit nominal +511 frequency coverage from the original LEO capture catalogue."""

import ast
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, experiment_output, require_run_context


COHORT = Path("reports/figures/2026_09_02_five_paired_pss_glrt/cohort-selection.json")
NATIVE_GLRT = Path("tools/analyze_0181_native25_fractional_glrt.py")
FIRST_CHANNEL_IF_HZ = 1_075_000_000
CHANNEL_SPACING_HZ = 250_000_000
CARRIER_SPACING_HZ = 234_375


def signed_bin(index):
    return (index + 512) % 1024 - 512


def carrier_if_hz(channel, index):
    midpoint = FIRST_CHANNEL_IF_HZ + (channel - 1) * CHANNEL_SPACING_HZ
    # The occupied channel midpoint lies half a subcarrier below bin zero.
    return midpoint + (signed_bin(index) + 0.5) * CARRIER_SPACING_HZ


def coverage(channel, tuned_hz, sample_rate_hz, rf_bandwidth_hz):
    target_hz = carrier_if_hz(channel, 511)
    baseband_hz = target_hz - tuned_hz
    limit_hz = min(sample_rate_hz, rf_bandwidth_hz) / 2
    return {
        "channel": channel,
        "tuned_if_hz": tuned_hz,
        "sample_rate_hz": sample_rate_hz,
        "rf_bandwidth_hz": rf_bandwidth_hz,
        "target_if_hz": target_hz,
        "target_baseband_hz": baseband_hz,
        "nominal_guard_hz": limit_hz - abs(baseband_hz),
    }


def capture_coverage(record, selected):
    stream = record["stream_25m"]
    center = stream["center_frequency_hz"]
    channel = 1 + round((center - FIRST_CHANNEL_IF_HZ) / CHANNEL_SPACING_HZ)
    return {
        "capture": record["session_id"],
        "selected_in_original_pss_study": record["session_id"] in selected,
        "bundle_uri": record["bundle_uri"],
        "manifest_digest": record["manifest_digest"],
        "stream": stream["stream_id"],
        "observed_density": stream["observed_density"],
        "continuity_gaps": stream["continuity_gap_count"],
        **coverage(channel, center, stream["sample_rate_hz"], stream["rf_bandwidth_hz"]),
    }


def main():
    require_run_context()
    sources = {rel.as_posix(): (EXTERNAL_LEO_ROOT / rel).read_bytes() for rel in (COHORT, NATIVE_GLRT)}
    catalogue = json.loads(sources[COHORT.as_posix()])
    selected = {record["session_id"] for record in catalogue["selected"]}
    captures = [capture_coverage(record, selected) for record in catalogue["eligible_ranked"]]
    names = {"CAPTURE_ID", "STARLINK_CHANNEL", "TUNED_CENTER_HZ", "SAMPLE_RATE_HZ", "RF_BANDWIDTH_HZ"}
    tree = ast.parse(sources[NATIVE_GLRT.as_posix()])
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                constants[name] = ast.literal_eval(node.value)
    independent = {
        "capture": constants["CAPTURE_ID"],
        "stream": "stream-0",
        **coverage(constants["STARLINK_CHANNEL"], constants["TUNED_CENTER_HZ"],
                   constants["SAMPLE_RATE_HZ"], constants["RF_BANDWIDTH_HZ"]),
    }
    output = {
        "schema": "starlink-leo-recording-coverage/v1",
        "sources_sha256": {rel: sha256(data).hexdigest() for rel, data in sources.items()},
        "cohort": captures,
        "separate_native_glrt_capture": independent,
        "limitations": "Nominal passband only: radio filter response, actual CFO and IQ are unmeasured here.",
    }
    output_dir = experiment_output("E071")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "leo_recording_coverage.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8",
    )
    print(f"{sum(row['nominal_guard_hz'] > 0 for row in captures)}/{len(captures)} cohort captures nominally cover +511")


if __name__ == "__main__":
    main()
