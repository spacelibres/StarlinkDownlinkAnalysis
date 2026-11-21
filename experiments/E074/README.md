# E074 · 高阶前缀预测与跨分组近重复

沿用 E070 的档案坐标，提取近全 PN 填充的高阶字。保留实收硬判、条件 PN 补尾及相应擦除。审计前缀近邻、仿射组合和相同两级头部的近邻帧，测量同一数据跨训练与留出的影响。

运行 `python tools/run_experiment.py audit_padding_parity_transfer --run-id <新编号> -- --nearby-diagnostic`。需要秦档案的同版范围缓存及 E028/E030/E036/E070 归档。省略脚本参数复现首轮计划。

- [设计](../../docs/plans/2026-11-21-padding-parity-transfer.md) · [报告](../../reports/E074/2026-11-21-高阶近重复与校验候选.md)
- [首轮](../../runs/v072-padding-parity-transfer/manifest.json) · [追加诊断](../../runs/v072-padding-parity-final/manifest.json) · [复跑](../../runs/v072-padding-parity-confirm/manifest.json)
