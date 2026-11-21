"""Render the held-band spectral and time-domain checks in the approved visual identity."""

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import resvg_py

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project.json").is_file())
sys.path.insert(0, str(ROOT))
from project_paths import VI_ROOT, experiment_output, require_run_context
from audit_ut_repeated_data import checked_artifact, read


NS = "http://www.w3.org/2000/svg"
SVG = "{" + NS + "}"


def apply_identity(path, colors):
    ET.register_namespace("", NS)
    tree = ET.parse(path)
    svg = tree.getroot()
    svg.set("width", "1400")
    svg.set("height", "680")
    decorative = ET.Element(SVG+"g", {"opacity": ".52", "aria-hidden": "true"})
    gradient = ET.SubElement(decorative, SVG+"linearGradient", {"id": "e040-header-orbit"})
    for offset, key, alpha in (("0", "secondary", "0"), (".51", "primary", ".28"), ("1", "accent", "0")):
        ET.SubElement(gradient, SVG+"stop", {"offset": offset, "stop-color": colors[key], "stop-opacity": alpha})
    ET.SubElement(decorative, SVG+"path", {"d": "M555 99 C635 14 830 -6 996 44 C1047 60 1071 71 1096 100",
                                          "stroke": "url(#e040-header-orbit)", "stroke-width": "1.4", "fill": "none"})
    figure = next(x for x in svg if x.get("id") == "figure_1")
    figure.insert(1, decorative)
    logo = ET.parse(VI_ROOT / "assets/horizontal-primary.svg").getroot()
    logo.set("x", "817")
    logo.set("y", "20")
    logo.set("width", "175")
    logo.set("height", str(175 * 180 / 810))
    if logo.get("data-color-mode") != "monochrome":
        for node in logo.iter():
            if node.get("data-stroke-role") == "planet":
                node.set("stroke", colors["planetOnLight"])
            elif node.get("fill", "").lower() == "#007bff":
                node.set("fill", colors["primaryOnLight"])
            elif node.get("fill", "").lower() == "#002044":
                node.set("fill", colors["ink"])
    svg.append(logo)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main():
    require_run_context()
    dependencies = {}
    run = "v038-wav-shifted-lines"
    stats = read(checked_artifact(run, "E040", "wav_shifted_lines.json", dependencies))
    data_path = checked_artifact(run, "E040", "wav_shifted_lines.npz", dependencies)
    colors = json.loads((VI_ROOT / "assets/palette-tokens.json").read_text(encoding="utf-8"))[0]["colors"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "text.color": colors["ink"],
                         "axes.facecolor": colors["card"], "axes.labelcolor": colors["ink"],
                         "axes.edgecolor": colors["line"], "axes.titleweight": "bold",
                         "ytick.color": colors["muted"], "xtick.color": colors["muted"],
                         "figure.facecolor": colors["surface"], "grid.color": colors["line"], "svg.fonttype": "path"})
    held, post = stats["heldout_blocks"], stats["post_blocks"]
    with np.load(data_path, allow_pickle=False) as data:
        delta = data["deltas"]
        p = data["profiles"][:, 0]
        times = data["centers_s"]
        offsets = data["offsets"][:, 0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.subplots_adjust(left=.084, right=.959, top=.739, bottom=.159, wspace=.28)
    fig.text(.055, .922, "Shifted spectral lines in the original IQ", fontsize=26, weight="bold")
    fig.text(.056, .865, "2.4 MS/s WAV  |  Predicted spacing 749.997 Hz  |  Negative and positive frequency bands",
             fontsize=13, color=colors["muted"])
    palette = [colors["primaryOnLight"], colors["planetOnLight"]]
    for j, (color, name) in enumerate(zip(palette, ("Negative band", "Positive band"), strict=True)):
        left.plot(delta, p[held, j].mean(axis=0), color=color, lw=2.1, label=name)
        left.plot(delta, p[post, j].mean(axis=0), color=color, lw=1.0, ls="--", alpha=.64)
        right.scatter(times[held], offsets[held, j], facecolor=color if not j else "none",
                      edgecolor=color, s=21 if not j else 37, marker="o")
    left.axvline(0, color=colors["muted"], lw=.8, ls=":")
    left.set(title="Spectral concentration versus offset", xlabel="Offset from the moving line's grid (Hz)",
             ylabel="Normalized spectral power", xlim=(-375, 375))
    left.legend(frameon=False, loc="upper right")
    right.axvline(13.635, color=colors["muted"], lw=.9, linestyle=":")
    right.text(13.51, -247, "End of event", fontsize=10, rotation=90,
               ha="right", va="top", color=colors["muted"])
    right.axhline(-260, color=colors["line"], lw=.9)
    right.set(title="Offsets estimated independently by band", xlabel="WAV time (s)",
              ylabel="Estimated line-grid offset (Hz)", xlim=(0, 14.1), ylim=(-305, -225))
    for ax in (left, right):
        ax.grid(alpha=.45, linewidth=.65)
        ax.set_axisbelow(True)
    fig.text(.055, .067, "Solid curves use held blocks during the event. Dashed curves use blocks after the event.",
             color=colors["muted"], fontsize=11)
    folder = experiment_output("E040").parent / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "wav-shifted-spectral-lines.svg"
    fig.savefig(path)
    plt.close(fig)
    apply_identity(path, colors)
    path.with_suffix(".png").write_bytes(resvg_py.svg_to_bytes(svg_string=path.read_text(encoding="utf-8"),
                                                              width=1680, dpi=96))


if __name__ == "__main__":
    main()
