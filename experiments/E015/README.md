# E015 · Code 9 的系统码与分组复核

用两个事先指定的整帧分组检验旧宽带录波中 Code 9 的系统编码。依据论文 §VI-D 和 CRC 支持的候选资源分配，使用沿袭的导频校正。保留原始判决、错位和交叉码号对照。两种划分共享同一捕获。

运行 `python tools/run_experiment.py validate_code9 --run-id <新编号> -- --unwrap-slope`。不加该选项可运行基线接收器。原计划见 `docs/plans/2026-11-20-code9-parity.md`，失败及后续修正见[实验报告](../../reports/E015/2026-11-20-Code9整帧校验与导频相位.md)。
