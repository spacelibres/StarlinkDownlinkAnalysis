# E143 · LEO 原工程导频检测器实测重放

以 `../leo-tracker-reduxredux-main` 为只读依赖，直接运行原工程的 `detect_pilot_methods`。使用 DSX01-S05 和 paired_RX 保存的原始样点，按样点哈希去重后为 10 个窗口。比较同一窗口的默认频偏范围和 ±600 kHz 范围，同时保存已知导频与循环移位对照的得分。重复运行属于确定性复算。样本来自同一条扫描，未作参数训练或独立统计推断。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
py -3.14 tools/run_experiment.py replay_leo_pilot_acquisition --run-id <新编号>
```

可用 `STARLINK_EXTERNAL_LEO_ROOT` 指向同一源码快照。[报告](../../reports/E143/2026-11-21-LEO原工程检测器重放.md) · [正式运行](../../runs/v140-leo-original-detector/manifest.json) · [复跑](../../runs/v140-leo-original-detector-confirm/manifest.json)
