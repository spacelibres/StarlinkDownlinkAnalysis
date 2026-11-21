# E046 · 偏移帧 Code 5 留出校验位

将偏移时轴候选 Code 5 的校验区按完整循环块屏蔽，用冻结稀疏矩阵计算预测。对比跨帧系统位先验、PN15 先验、错位及强帧，并评估完整码字译码。

- [实验计划](../../docs/plans/2026-11-21-ut-offset-code5-parity.md)
- [实验报告](../../reports/E046/2026-11-21-偏移帧Code5留出校验.md)
- [归档清单](../../runs/v044-ut-offset-code5-parity/manifest.json)

复跑命令 `python tools/run_experiment.py audit_ut_offset_code5_parity --run-id <新编号>`。
