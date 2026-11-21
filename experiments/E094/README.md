# E094 · Code 5 首条记录的位置与速度

从通过外层校验与目录 CRC24 的首码字提取三个 binary64 和三个 binary32 值。用独立实录、相对帧计数、位置差分、旋转坐标系动力学和错误读法对照检验字段解释，再计算 LEO 与原 WAV 的理想位覆盖。

运行 `python tools/run_experiment.py audit_code5_orbit_state --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 均设为 1。比较来源包含 LEO Tracker 原工程的坐标变换代码，默认以 `../leo-tracker-reduxredux-main` 为根，可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定相同的源文件树。

[报告](../../reports/E094/2026-11-21-Code5地固位置与速度.md) · [运行](../../runs/v092-code5-orbit-state/manifest.json) · [复跑](../../runs/v092-code5-orbit-state-confirm/manifest.json)
