# E038 · Code 5 到 MCS 6 的相位转移与扩展记录

用前置 Code 5 校验独立估计边缘载波相位，并检验 MCS 6 中与 `E` 相邻的标签与停止位置之后的固定锚。方案见[实验计划](../../docs/plans/2026-11-21-mcs6-record-and-phase-transfer.md)。

运行 `python tools/run_experiment.py audit_qin_mcs6_transfer --run-id <新编号>`，读取 E036 同一档案修订和已有范围缓存。

`python tools/run_experiment.py verify_qin_mcs6_anchor_edge --run-id <新编号>` 从两次已归档观测复核被屏蔽的边缘位。[实验报告](../../reports/E038/2026-11-21-MCS6扩展记录与跨区相位.md)
