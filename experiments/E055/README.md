# E055 · LEO Tracker 原工程数据与双路 IQ 复核

直接只读检验 `../leo-tracker-reduxredux-main` 中的文件类型、采集来源和双路 IQ。用 DSX01-S05 原记录的频率与时序信息，检查相对频漂在整帧留出区的表现。

- [实验计划](../../docs/plans/2026-11-21-leo-paired-replay.md) · [报告与原工程对应表](../../reports/E055/2026-11-21-LEO原工程数据与双路复核.md)
- [外部目录清点复跑](../../runs/v053-leo-live-inventory/manifest.json) · [数据复核](../../runs/v053-leo-pair-final/manifest.json) · [重复运行](../../runs/v053-leo-pair-confirm/manifest.json)
- [首轮 JSON 序列化失败](../../runs/v053-leo-pair-replay/manifest.json)

`python tools/run_experiment.py audit_leo_pairs_live --run-id <新编号>`。默认从 `../leo-tracker-reduxredux-main` 读取，另一个目录可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定。实验检查四个外部 IQ/选择文件的哈希与项目副本一致。
