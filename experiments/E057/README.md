# E057 · LEO 双路逐符号预测

从 LEO Tracker 原工程读取成对 IQ。固定导频给出的时轴及频偏分支，逐符号检验共同可见的非导频子载波，同时用相同估计器测试已知导频。

- [实验设计](../../docs/plans/2026-11-21-leo-paired-slot-transfer.md) · [报告](../../reports/E057/2026-11-21-LEO双路逐符号预测.md)
- [首轮](../../runs/v055-leo-paired-slots/manifest.json) · [全部共同导频](../../runs/v055-leo-all-pilot-controls/manifest.json) · [复跑](../../runs/v055-leo-all-pilot-confirm/manifest.json)

`python tools/run_experiment.py audit_leo_paired_slots --run-id <新编号>`。外部目录默认为 `../leo-tracker-reduxredux-main`，也可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定。
