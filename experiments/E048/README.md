# E048 · 三类录波的残差预测与逐帧相位

原 WAV 检查两侧残差，NPZ 检查同一帧内独立周期组的变化，宽带使用已测二元尾段校验逐帧相位的处理。NPZ 使用宽带冻结模板复验参考相位，保留未通过的帧和窗口。

- [实验计划](../../docs/plans/2026-11-21-cross-capture-residual-transfer.md)
- [实验报告](../../reports/E048/2026-11-21-三类录波残差与逐帧相位.md)
- [最终运行清单](../../runs/v046-cross-capture-phase-final/manifest.json)

复跑用 `python tools/run_experiment.py audit_residual_transfer --run-id <新编号>`。
