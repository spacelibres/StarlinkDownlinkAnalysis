# E124 · Qin 全档案的 MCS 85 记录与计数

在 1,009 帧 Qin 实测档案中重新提取每一处 MCS 85 分配。冻结 E052 的标号、PN15、20 字节窗口、检测门槛和三种标签到间距的关系。拼接各码字的系统区，在原分配内测试后继位置。保持最近星座点判决原样，并用 E081 的精确格点法记录哪些位具备观察支持。错 PN15 一位、置换标签、固定长度和字节偏移提供对照。

沿用 E077 整组划分及种子 `2026112177`，留出另排除 E034 已绑定到 250 MS/s 原始录波的整组。全档案初筛发现 `03 e9` 和 1,512 字节间距的关系。该候选单独保留，受检组已经用于发现。观察每帧同余位置和固定步长计数，不把同组重叠配对视作独立验证。

LEO 投影使用 E122 已校验的频窗元数据，逐位映射偏移 4–7 的标签与计数坐标。外部原工程和本机 Qin 页缓存只读。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs85_archive_records --run-id <新编号>
```

[报告](../../reports/E124/2026-11-21-MCS85全档案标签与计数.md) · [当前源码复跑](../../runs/v122-mcs85-archive-records-verified/manifest.json) · [最终结果](../../runs/v122-mcs85-archive-records-final/manifest.json) · [确认](../../runs/v122-mcs85-archive-records-final-confirm/manifest.json)
