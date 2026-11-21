# E053 · MCS 85 计数候选与公开帧格式

用 E052 的固定记录位置重取交叉 32QAM 软符号，检查同标签记录中偏移 6–7 的候选序号。保留原始字节、进位歧义及 IPv4、IPv6 和 Ethernet 的有限否定结果。

- [实验计划](../../docs/plans/2026-11-21-mcs85-counter-soft.md)
- [实验报告](../../reports/E053/2026-11-21-MCS85计数与公开帧格式.md)
- [有效运行](../../runs/v051-mcs85-counter-final/manifest.json) · [复跑](../../runs/v051-mcs85-counter-replay/manifest.json)
- [首轮档案硬判不一致的记录](../../runs/v051-mcs85-counter/manifest.json)

复跑用 `python tools/run_experiment.py audit_mcs85_counter --run-id <新编号>`。
