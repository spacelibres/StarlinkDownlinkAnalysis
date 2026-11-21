# E052 · MCS 85 系统区的跨码字记录与长度

在 250 MS/s 录波的 MCS 85 系统区，对独立的检测位和验证位评分，检验两个实测字节能否预测下一条记录的位置。冻结模板和间距规则，再转移到 Qin 的频域档案。

- [实验计划](../../docs/plans/2026-11-21-mcs85-record-spacing.md)
- [实验报告](../../reports/E052/2026-11-21-MCS85记录长度.md)
- [原 IQ 分组运行](../../runs/v050-mcs85-spacing/manifest.json)
- [Qin 档案转移运行](../../runs/v050-mcs85-qin/manifest.json)
- [原 IQ 复跑](../../runs/v050-mcs85-spacing-replay/manifest.json) · [档案转移复跑](../../runs/v050-mcs85-qin-replay/manifest.json)

复跑脚本 `audit_mcs85_records` 和 `transfer_mcs85_records_qin` 均通过 `python tools/run_experiment.py <脚本名> --run-id <新编号>` 运行。转移脚本读取已归档的冻结原 IQ 结果。
