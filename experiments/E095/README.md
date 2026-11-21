# E095 · Code 5 的 GPS 时间标记与同期轨道

从通过校验的首码字提取整秒、纳秒和 15 秒区间候选标记。用 Qin 作者的 TOA 与多普勒、固定地面参考位置和历史 TLE 检验读法。复用 LEO Tracker `src/leo/sky/frames.py` 中的 TEME 到地固变换，默认根目录为 `../leo-tracker-reduxredux-main`。也可使用 `STARLINK_EXTERNAL_LEO_ROOT` 指向同一版本的原文件。

主实验运行 `python tools/run_experiment.py audit_code5_gps_epoch --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 设为 1。主实验读取已归档的测量列与 TLE，不访问网络。若本机有 E034 的两片源文件，可用 `extract_qin_time_observables` 复核源列提取。两次主运行使用同一测量输入。

[完整报告](../../reports/E095/2026-11-21-Code5时间标记与同期轨道.md) · [TOA 与 Doppler 源列](../../runs/v093-qin-time-source/manifest.json) · [运行](../../runs/v093-code5-gps-epoch/manifest.json) · [复跑](../../runs/v093-code5-gps-epoch-confirm/manifest.json)
