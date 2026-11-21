# E054 · StarLoc、原始 WAV 与 LEO 选段的窄带对照

使用公共三路同步录波，检验约 43.945 kHz 梳谱、共享频漂以及 1.333 ms 相关峰在不同频窗的位置和检出条件。

- [实验计划](../../docs/plans/2026-11-21-starloc-narrowband.md)
- [实验报告](../../reports/E054/2026-11-21-StarLoc九音与帧周期.md)
- [原始区间接收记录](../../data/raw/starloc_obs0/selection.json) · [替代下载入口复查](../../provenance/starloc_obs0_range_recheck.json)
- [运行](../../runs/v052-starloc-final/manifest.json) · [复跑](../../runs/v052-starloc-replay/manifest.json)

使用 `python tools/run_experiment.py audit_starloc_narrowband --run-id <新编号>`。只有在本地没有归档数据时，才需要用 `import_starloc_ranges.py` 取相同的区间。
