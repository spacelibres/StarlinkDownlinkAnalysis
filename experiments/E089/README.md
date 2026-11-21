# E089 · LEO 分数样点定时与跨帧导频

在两份原始 LEO NPZ 的十个唯一窗口上，把分数样点帧间隔与正、反向整数舍入比较。三种定时均使用同一批、落在相同 OFDM 符号内的样点。对照逐帧逐载波自由增益和跨帧共享通道，再检验冻结 60 位序列带来的单独预测增量。

用 `py -3.14 tools/run_experiment.py audit_leo_fractional_timing --run-id <新编号>` 复跑。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 均设为 1。默认只读原工程目录，可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定相同输入。

[报告](../../reports/E089/2026-11-21-LEO分数样点定时与留出预测.md) · [实测](../../runs/v087-leo-fractional-timing/manifest.json) · [复跑](../../runs/v087-leo-fractional-timing-confirm/manifest.json)
