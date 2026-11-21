# E028 · MCS 66 与 70 编码审计

用 250 MS/s 录波的九个完整帧构建候选资源字，并将 66/70 与冻结的旧校验关系作转移比较。原始符号和细化符号均存档，选择位序时只使用拟合帧。

命令 `python tools/run_experiment.py extract_ut_qam --run-id <新编号>` 和 `python tools/run_experiment.py audit_ut_qam_fec --run-id <新编号>`。[实验报告](../../reports/E028/2026-11-21-MCS66和70冻结编码转移.md)
