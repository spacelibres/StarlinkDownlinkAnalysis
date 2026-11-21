# E051 · 从 PN15 填充停止点定位同星座分界

在 250 MS/s 实测 IQ 的完整资源格中，先从八种已知格式的 PN15 匹配变化定位候选码字，再与 H2 描述符比较。短后窗保留了 33 个候选，全部与预测的分配末端逐资源对齐，包含 19 个同星座内部边界。

- [实验计划](../../docs/plans/2026-11-21-blind-pn-boundaries.md)
- [实验报告](../../reports/E051/2026-11-21-PN15填充与同星座分界.md)
- [修正后的运行清单](../../runs/v049-blind-pn-corrected/manifest.json)
- [复跑清单](../../runs/v049-blind-pn-replay/manifest.json)
- [初次运行，H2 对照有实现错误](../../runs/v049-blind-pn-final/manifest.json)

复跑用 `python tools/run_experiment.py audit_blind_pn_boundaries --run-id <新编号>`。
