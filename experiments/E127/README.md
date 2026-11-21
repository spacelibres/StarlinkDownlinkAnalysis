# E127 · 跨码率校验转移与第六族软译码

从 E085 和 E120 的冻结矩阵比较系统项及奇偶项。MCS 66 的第五族和 MCS 70 的第一族之间，所有系统位在串行坐标上恰好相差 512 位，奇偶项完全一致。以此为线索，用同一对齐规则转移 MCS 70 的第二族。记录系统项超界及奇偶项落入目标系统区而无法映射的情形。

新提取的 Qin 中间码字来自所有至少三个字的可用高阶分配。每个分配按零基编号取 `count // 2`，原首字和末字只作为另一个证据层。使用 E077 固定组与 E124 的同源组排除。留出矩阵比较只评分完整实收行，并以错位前后都可见的坐标评分。中间字来自相同采集会话和档案组。原始 250 MS/s 的 QAM16 与交叉 32QAM 检验分别使用 E028、E030 的保留帧。其他 MCS 和旧三奇偶项校验迁移均为探索对照。

用 E125 的既有方法与轮数比较五族、六族，以及保留五族再加一族错位的 MCS 85 译码。该比较不重新选星座、可靠性尺度、校准帧、记录位置和计数对。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_cross_rate_checks --run-id v125-cross-rate-checks
python tools/run_experiment.py audit_six_family_mcs85 --run-id <新编号>
```

[报告](../../reports/E127/2026-11-21-跨码率校验与第六族.md) · [校验运行](../../runs/v125-cross-rate-checks/manifest.json) · [复跑](../../runs/v125-cross-rate-checks-confirm/manifest.json) · [软译码](../../runs/v125-six-family-mcs85/manifest.json) · [复跑](../../runs/v125-six-family-mcs85-confirm/manifest.json)
