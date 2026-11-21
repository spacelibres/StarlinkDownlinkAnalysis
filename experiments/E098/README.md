# E098 · 从限带实测 IQ 读出 Code 5 帧时钟

将 480 MS/s 实测宽带 IQ 按 LEO 上边缘的标称中心限带，输出 25 MS/s 复样点。用限带后的 PSS、SSS 和导频收帧，从观测到的系统位还原 E097 的时间。压力测试包括边缘滚降、调谐偏移、错误极性和擦除 511。

运行 `python tools/run_experiment.py audit_code5_narrow_waveform_clock --run-id <新编号>`。归档运行设 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。外部 LEO 源只读取固定配置文件的哈希，目录可由 `STARLINK_EXTERNAL_LEO_ROOT` 指定。

[报告](../../reports/E098/2026-11-21-限带IQ解出Code5帧时钟.md) · [直接读数初次运行](../../runs/v096-code5-clock-native25/manifest.json) · [最终运行](../../runs/v096-code5-clock-cadence/manifest.json) · [最终复跑](../../runs/v096-code5-clock-cadence-confirm/manifest.json)
