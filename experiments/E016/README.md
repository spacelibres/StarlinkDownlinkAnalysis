# E016 · Code 21 的循环块生成关系

使用时间连续的导频相位校正与固定 QPSK 标号。按整帧划分评估 Code 21 的系统生成关系，比较 Code 9 的两种直接指数转移。

运行 `python tools/run_experiment.py validate_code21 --run-id <新编号> -- --training-convention-bank`。省略最后一个选项可重现初始标号的矛盾。实验计划见 `docs/plans/2026-11-20-code21-qc.md`，结果见[实验报告](../../reports/E016/2026-11-20-Code21循环编码与单比特校验.md)。
