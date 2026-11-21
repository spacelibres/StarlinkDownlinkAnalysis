# E064 · LEO Tracker 导频相位的整块留出检验

用原工程的 FrameClock 和保存的 IQ 检验八根下边缘导频。对照最近邻相位、其余七根的加权线性相位，以及逐根独立估相。校准和留出按连续符号块分配。

- [实验计划](../../docs/plans/2026-11-21-leo-pilot-phase-transfer.md) · [报告](../../reports/E064/2026-11-21-LEO导频相位留出检验.md)
- [最近邻首轮](../../runs/v062-leo-pilot-phase/manifest.json) · [首轮复跑](../../runs/v062-leo-pilot-phase-confirm/manifest.json)
- [加入七载波对照](../../runs/v062-leo-pilot-phase-global/manifest.json) · [复跑](../../runs/v062-leo-pilot-phase-global-confirm/manifest.json) · [逐帧数值](../../runs/v062-leo-pilot-phase-global/artifacts/experiments/E064/results/leo_pilot_phase_transfer.json)

`py -3.14 tools/run_experiment.py audit_leo_pilot_phase_transfer --run-id <新编号>`。需要安装 NumPy 及原工程可导入依赖，外部源码和 IQ 位置可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定。脚本核对 E056 的输入文件和去重指纹。
