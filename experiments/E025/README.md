# E025 · 二元尾段边界和周期

在 250 MS/s 完整帧中，先独立比较星座寻找尾段，再用互补整帧组识别和验证重复序列。冻结后的尾段起点用于比较头部资源长度模型。受截断的候选帧 9 单列。

运行 `python tools/run_experiment.py audit_ut_binary_tail --run-id <新编号>`。[研究记录](../../reports/E025/2026-11-21-250MSps二元尾段和周期.md) · [实验计划](../../docs/plans/2026-11-21-ut-binary-tail.md)
