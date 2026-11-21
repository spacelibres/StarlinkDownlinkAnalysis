# E092 · Code 5 时变位段及窄带 CRC 信息量

以 E091 已验证的目录和码字为输入，用独立到达时轴及 E077 的整组留出检验首条记录的时变量。对照离轴条目、内容计数、两份原始宽带、原始 WAV 与 LEO 频窗，并验证首条记录尾部和下一条的固定前缀。

运行 `python tools/run_experiment.py audit_code5_record_trajectories --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 设为 1。

[报告](../../reports/E092/2026-11-21-Code5时变位段与CRC可辨识性.md) · [运行](../../runs/v090-code5-trajectories/manifest.json) · [复跑](../../runs/v090-code5-trajectories-confirm/manifest.json)
