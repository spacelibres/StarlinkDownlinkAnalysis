# E060 · 载荷准循环常数投影的软判检验

保留 16QAM 符号内部的位相关，检验候选折叠关系能否从拟合帧转移到整个留出帧。已知 PN15 校准软值，Code 5 提供实测正对照。

- [实验设计](../../docs/plans/2026-11-21-qc-soft-folds.md) · [报告](../../reports/E060/2026-11-21-准循环折叠软判.md)
- [首轮](../../runs/v058-qc-soft-folds/manifest.json) · [扩展](../../runs/v058-qc-soft-folds-expanded/manifest.json) · [复跑](../../runs/v058-qc-soft-folds-confirm/manifest.json)

`python tools/run_experiment.py audit_qc_soft_folds --run-id <新编号>`。归档复数观测、PN15 模型及 Code 5 矩阵的哈希随数值结果保存。
