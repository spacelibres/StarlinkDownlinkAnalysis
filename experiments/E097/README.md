# E097 · Code 5 冗余帧时钟及 LEO 频窗重建

用精确整数关系核对首条记录的 23 位时钟修正、帧计数和 15 秒区间锚。按 LEO 历史 25 MS/s 频窗投影真实宽带判决，擦除原生 GPS 时间位，检验绝对时刻和区间内时刻能否从其他字段重建。

运行 `python tools/run_experiment.py audit_code5_clock_correction --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 设为 1。输入均为本地冻结证据，含 LEO 原工程轨道与时序综述的只读来源摘要。

[报告](../../reports/E097/2026-11-21-Code5冗余帧时钟.md) · [运行](../../runs/v095-code5-clock-final/manifest.json) · [复跑](../../runs/v095-code5-clock-final-confirm/manifest.json) · [无锚位补全的初次运行](../../runs/v095-code5-clock-correction/manifest.json)
