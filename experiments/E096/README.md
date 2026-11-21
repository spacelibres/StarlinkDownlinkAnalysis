# E096 · Code 5 空口卫星编号及同期目录排名

以 Qin 的作者标签和上一版同期轨道证据定位候选位。冻结读取方式后，从独立 480 MS/s 实录读出另一个编号。按解出的绝对时间和地固状态排名两份历史 TLE 目录，并另外计算原始 WAV 与 LEO 频窗的编号可辨识位数。

运行 `python tools/run_experiment.py audit_code5_fleet_identity --run-id <新编号>`。归档设 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。历史目录的原始 ZIP 和来源摘要已冻结，主实验离线运行。比较调用哈希与 E095 相同的 LEO Tracker `src/leo/sky/frames.py`，外部根可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定。

[报告](../../reports/E096/2026-11-21-Code5空口卫星编号.md) · [历史目录来源](../../provenance/code5_identity_orbit_catalogs.json) · [两目录运行](../../runs/v094-code5-fleet-full/manifest.json) · [复跑](../../runs/v094-code5-fleet-full-confirm/manifest.json) · [仅 2022 全目录的首轮](../../runs/v094-code5-fleet-identity/manifest.json)
