# E045 · 偏移帧的 H2 起点与系统区重复

用 E044 的分帧复数软观测验证 H1 之后的资源布局。从互补整帧组恢复 H2 首字节及起点，用全相位搜索和留出区间验证三个候选 Code 5 系统区的 PN15，并保留不能由该模型解释的重复。

- [实验计划](../../docs/plans/2026-11-21-ut-offset-h2-and-systematic.md)
- [实验报告](../../reports/E045/2026-11-21-偏移帧H2定位与系统区PN15.md)
- [归档清单](../../runs/v043-ut-offset-h2-final/manifest.json)

复跑命令 `python tools/run_experiment.py audit_ut_offset_h2_systematics --run-id <新编号>`。
