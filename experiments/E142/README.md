# E142 · Code 5 窄带记录边界搜索

给定首批 Code 5 的资源起点与系统位扰码，撤去 E141 使用的已知记录起点和类型。新版在 Qin 实测判决与 LEO 标称 25 MS/s 的频窗交集上逐字节搜索。旧版从 480 MS/s 样点生成三个内部频窗并重新同步，逐字节检测局部长度标签。结果包含错标记、反极性、错误标签以及 LEO 原生观测可以覆盖的位数。两次运行是确定性复算，三个内部频窗共享同一次录波。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_blind_extents --run-id <新编号>
```

[完整报告](../../reports/E142/2026-11-21-Code5窄带记录起点搜索.md) · [最终运行](../../runs/v141-code5-blind-extents-final/manifest.json) · [复跑](../../runs/v141-code5-blind-extents-confirm/manifest.json)
