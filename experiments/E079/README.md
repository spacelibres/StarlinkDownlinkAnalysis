# E079 · Code 5 帧计数与 LEO 频窗

用独立测时检验 Code 5 首字中的帧计数，比较九个离轴码字及档案中的重复内容。再用 LEO Tracker 原工程选段及历史录波清单，算出哪些频窗具有识别同一字段的理想信息量。

- `python tools/run_experiment.py audit_code5_frame_count --run-id <新编号> -- --off-grid-links`
- `python tools/run_experiment.py audit_leo_code5_projection --run-id <新编号>`

帧计数使用 E012、E014、E036、E073、E074、E077 和 E078 的冻结产物。频窗实验使用 E071 的原工程目录审计和本次帧计数的八类偏移，复核原工程两份选段及两份选择清单的哈希。默认外部路径为 `../leo-tracker-reduxredux-main`，也可使用 `STARLINK_EXTERNAL_LEO_ROOT`。

- [计划](../../docs/plans/2026-11-21-code5-frame-count.md) · [报告](../../reports/E079/2026-11-21-Code5帧序计数与离轴帧.md)
- [帧计数运行](../../runs/v077-code5-time-links/manifest.json) · [帧计数复跑](../../runs/v077-code5-time-links-confirm/manifest.json)
- [LEO 投影运行](../../runs/v077-leo-code5-projection/manifest.json) · [LEO 投影复跑](../../runs/v077-leo-code5-projection-confirm/manifest.json)
