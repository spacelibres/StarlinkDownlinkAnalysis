# E019 · Code 28 缩短编码

两个整帧划分的完整矩阵拟合均未通过。保留原始观测，继续检验字界和线性投影。

入口 `python tools/run_experiment.py validate_code28 --run-id <新编号>` 和 `python tools/run_experiment.py audit_code28_boundaries --run-id <新编号>`。后者读取封存的初次运行。结果见[实验报告](../../reports/E019/2026-11-20-Code28字界与块奇偶验证.md)。
