# E076 · 三重观测融合与 Code 5 擦除校验

使用同一档案内的相同两级头部和前缀注册码字，轮换留出第三次观测。已知 PN15 提供额外位置对照。以冻结 Code 5 生成矩阵验证擦除重建，分别保存实收与矩阵补全。

运行 `python tools/run_experiment.py audit_repeat_consensus --run-id <新编号> -- --code5-control-partial`。复用固定版本的秦档案范围缓存和 E012/E028/E030/E036/E070 的归档。

- [实验设计](../../docs/plans/2026-11-21-repeat-consensus.md) · [报告](../../reports/E076/2026-11-21-三重观测与擦除重建.md)
- [首轮](../../runs/v074-repeat-consensus/manifest.json) · [全可见 Code 5](../../runs/v074-repeat-consensus-final/manifest.json)
- [最终运行](../../runs/v074-repeat-consensus-erasure-final/manifest.json) · [复跑](../../runs/v074-repeat-consensus-erasure-confirm/manifest.json)
