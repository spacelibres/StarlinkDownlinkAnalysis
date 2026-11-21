# E071 · LEO 配对录波档案与 +511 标称覆盖

从外部原工程的配对录波清单复算上边缘 +511 的载波覆盖，校正 E063 报告中把循环 FFT 索引当成频率差的错误。输入为只读本地文件，数据根目录可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定。

运行 `python tools/run_experiment.py audit_leo_recording_coverage --run-id <新编号>`。产物保存逐录波覆盖、原始清单位置及哈希。已登记两次复算，源 CI16 仍需通过原归档取得。

- [计划](../../docs/plans/2026-11-21-leo-recording-coverage.md) · [报告](../../reports/E071/2026-11-21-LEO配对录波与载波覆盖.md)
- [运行](../../runs/v069-leo-recording-coverage/manifest.json) · [复跑](../../runs/v069-leo-recording-confirm/manifest.json)
