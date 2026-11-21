# E085 · MCS 66 窄相位窗校验

用前缀不超过 512 位的八个末字对同旋转校验区作差。按局部相位窗寻找一个低块加三个高块的稀疏关系，扣除旧矩阵的张成空间。用校准末字恢复系统项，在档案留出及原始 IQ 的硬判上测零综合，并用冻结规则比较单擦除补值。

运行 `python tools/run_experiment.py recover_mcs66_phase_window_check --run-id <新编号>`。结果包括打包矩阵、逐行支持、搜索窗口、实收对照与擦除候选。

[报告](../../reports/E085/2026-11-21-MCS66第五族校验.md) · [运行](../../runs/v085-mcs66-window-check/manifest.json) · [复跑](../../runs/v085-mcs66-window-check-confirm/manifest.json)
