# E030 · PN15 和交叉 32QAM 标号

使用 E025 的宽带资源长度模型提取其余格式的候选字。带入 E029 已冻结的 PN 相位与字节位序，只从 MCS 85 的拟合窗口求交叉 32QAM 点到五位标签的映射。

运行 `python tools/run_experiment.py extract_ut_cross32 --run-id <新编号>` 和 `python tools/run_experiment.py audit_ut_cross32_labels --run-id <新编号>`。[实验报告](../../reports/E030/2026-11-21-交叉32QAM标号与PN15边界.md)
