# E024 · 窄带第二头部投影

冻结两份公开宽带录波中的第二头部，按用户 WAV 和 NPZ 的条件频窗建立理想载波擦除掩码。比较完整带宽、各频窗和假想联合观测的秩，保留重新计算 CRC 后仍无法区分的替代字。

- `python tools/run_experiment.py validate_ut_capture --run-id <新编号>` 可重跑 250 MS/s 校验。E024 使用归档运行 `v022-ut-replay` 的结果。
- `python tools/run_experiment.py audit_h2_projection --run-id <新编号>` 运行投影审计。
- [实验报告](../../reports/E024/2026-11-21-250MSps录波与窄带头部投影.md) · [计划](../../docs/plans/2026-11-21-narrowband-h2-projection.md)
