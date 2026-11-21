# E100 · Code 5 多记录目录、分配终点与 LEO 频窗

检验 E091 的首条记录边界能否推广到完整目录，同时核对各长度和 H1 分配的关系。目录语法使用字节输入，CRC24、信道校验、实际填充和 LEO 频窗的线性秩分别计算。

输入为冻结的 Qin H1 与首 Code 5 字、Qin 可验证的频域页缓存、480 MS/s 与 250 MS/s 硬判证据，以及 LEO 的实录频偏和标称调谐。Qin 沿用 E077 的整组划分及种子 `2026112177`。此次没有拟合新随机划分。早期探索样本来自归档，当前留出用于冻结语法的复算。各帧、整组及输入哈希都在产物中保留。LEO 原工程只读。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_directory_records --run-id <新编号>
```

本机需已有 Qin 页缓存。脚本按 E091 的已登记哈希复验实际访问的页，无网络下载。LEO 工程目录可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指向相同输入树。三条本机 LEO 证据入口和物理来源见 [E071](../E071/README.md)、[E079](../E079/README.md)、[E099](../E099/README.md)。

[实验报告](../../reports/E100/2026-11-21-Code5多记录目录与LEO覆盖.md) · [初轮](../../runs/v098-code5-records/manifest.json) · [最终运行](../../runs/v098-code5-records-final/manifest.json) · [复跑](../../runs/v098-code5-records-confirm/manifest.json)
