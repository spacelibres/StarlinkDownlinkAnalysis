# E050 · 帧内分配次序与实际星座跳变

从两份宽带 IQ 重取软符号，在整帧上检测调制跳变，再与 CRC 已验证的头部预测对比。用总长度相同的顺序交换作对照，并对同一网格做窄带投影。

- [实验计划](../../docs/plans/2026-11-21-internal-allocation-boundaries.md)
- [实验报告](../../reports/E050/2026-11-21-帧内调制与分配次序.md)
- [最终运行清单](../../runs/v048-internal-allocation-final/manifest.json)

复跑用 `python tools/run_experiment.py audit_internal_allocations --run-id <新编号>`。
