# Interim manuscript and reproducibility note

The manuscript uses the IEEE journal article structure and numbered citation style. It is an interim English draft with a Chinese abstract and contribution table. Author names, affiliations, funding declarations and a target venue are pending confirmation. The scientific cut-off is `v0.0.144`, commit `1eabe0fa37013d8069ac5effa3ab3f305c67bf18`.

- [Paper PDF](starlink-ku-downlink-interim.pdf)
- [Editable English manuscript](manuscript.md)
- [中文摘要与公开研究贡献对照](中文摘要与贡献对照.md)
- [IEEE journal article structure](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/structure-your-article/)

## Traceability

| Manuscript result | Experiment or archived measurement |
|---|---|
| Capture provenance, overlapping Qin/UT frames and H1/H2 | [E010](../../experiments/E010/README.md), [E014](../../experiments/E014/README.md), [E034](../../experiments/E034/README.md), [E036](../../experiments/E036/README.md) |
| WAV periodicity, event edge, repeat structure and StarLoc control | [E042](../../experiments/E042/README.md), [E047](../../experiments/E047/README.md), [E054](../../experiments/E054/README.md), [E090](../../experiments/E090/README.md), [E116](../../experiments/E116/README.md) |
| Code 5 counter, CRC-24, ECEF, time, number and clock | [E079](../../experiments/E079/README.md), [E091](../../experiments/E091/README.md), [E094](../../experiments/E094/README.md), [E095](../../experiments/E095/README.md), [E096](../../experiments/E096/README.md), [E097](../../experiments/E097/README.md) |
| Gain, coding, holdouts and equivalent matrices | [E128](../../experiments/E128/README.md), [E131](../../experiments/E131/README.md), [E132](../../experiments/E132/README.md) |
| Outer records, scheduling rules, transforms and extents | [E134](../../experiments/E134/README.md), [E137](../../experiments/E137/README.md), [E138](../../experiments/E138/README.md), [E139](../../experiments/E139/README.md), [E140](../../experiments/E140/README.md), [E141](../../experiments/E141/README.md), [E142](../../experiments/E142/README.md) |
| Native and projected narrowband evidence | [E130](../../experiments/E130/README.md), [E143](../../experiments/E143/README.md), [E144](../../experiments/E144/README.md) |
| Hard and soft Code 28 witnesses | [E145](../../experiments/E145/README.md), [E146](../../experiments/E146/README.md) |

Each experiment entry points to its commands and archived runs. Run manifests record direct artifact hashes or page hashes, and the applicable seeds, assignments, erasure masks and controls are in the documented results. The legacy and Qin capture families are distinct. The 250 MS/s replay adds raw-sample checks and overlaps with Qin; native narrowband exports and synthetic controls retain their own provenance.

## Regenerate the PDF

Use Python with ReportLab 5.1.0. From the project root, run:

```powershell
python reports/interim-paper/render.py
```

The layout follows IEEE-style journal structure and uses a deterministic ReportLab renderer. No specific journal's submission template was selected. The rendering script uses the three held-control TOA RMS measurements documented in E095 for Figure 1. The paper neither changes experiment data nor reruns the experiments when rendered.
