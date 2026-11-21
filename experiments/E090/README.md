# E090 · 原始 WAV 周期分量的终止

冻结 E040–E042 的周期时轴与频谱形状，用同一份原始 WAV 比较两侧相干模板、移动细线和固定接收频带的终止时刻。保留四种分箱相位、同谱随机相位对照及对事件前后分层的块残差抽样。

运行 `python tools/run_experiment.py audit_wav_frame_termination --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 均设为 1。

[报告](../../reports/E090/2026-11-21-WAV周期分量终止.md) · [首轮分组](../../runs/v088-wav-frame-termination/manifest.json) · [分层修正](../../runs/v088-wav-frame-cut-stratified/manifest.json) · [复跑](../../runs/v088-wav-frame-cut-confirm/manifest.json)
