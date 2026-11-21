# E067 · 全档案尾段相位与头部分配

从 Qin 的 1,009 帧档案和另一份 480 MS/s 原始录波测量终端结构。先完成信号测量，再读取头部结果，比较逐字和整段取整。不合格帧及对照一并保留。

- [实验计划](../../docs/plans/2026-11-21-full-tail-phase-audit.md) · [实验报告](../../reports/E067/2026-11-21-全档案尾段相位与头部分配.md)
- [首轮](../../runs/v065-full-tail-phase/manifest.json) · [最终](../../runs/v065-full-tail-phase-final/manifest.json) · [复跑](../../runs/v065-full-tail-phase-confirm/manifest.json)

`python tools/run_experiment.py audit_full_tail_phase --run-id <新编号>`。需要 h5py 和原工程已经保存的 Qin 数据分块缓存。脚本只读完整页并保存实际所读区段的指纹。归档的 [JSON](../../runs/v065-full-tail-phase-final/artifacts/experiments/E067/results/tail_phase_audit.json) 和 [末段样点 NPZ](../../runs/v065-full-tail-phase-final/artifacts/experiments/E067/results/tail_observations.npz) 可用于脱离缓存复核主判据。
