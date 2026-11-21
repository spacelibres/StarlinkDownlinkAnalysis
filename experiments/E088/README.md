# E088 · MCS 66 部分软译码

用 E084 与 E085 的冻结矩阵，比较无校验、四族、五族及系统项错移五族。档案使用校准组先验和原有的 22 个留出组，原始 IQ 使用 16QAM 复观测的边际 LLR。已知 PN 只用于校准或评分。结果保留擦除、近重复对照与逐位判决掩码。

运行 `python tools/run_experiment.py audit_mcs66_partial_decoder --run-id <新编号>`。

[报告](../../reports/E088/2026-11-21-MCS66部分软译码.md) · [首轮](../../runs/v086-mcs66-decoder-first/manifest.json) · [补充对照](../../runs/v086-mcs66-decoder/manifest.json) · [复跑](../../runs/v086-mcs66-decoder-confirm/manifest.json)
