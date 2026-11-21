# E134 · 外层记录长度列表和跨 MCS 的 CRC24 目录

以 E104 已通过目录校验的长度为参照，从实收外层头读取首项和后续 12 位长度项，保留头内擦除及未知标志。宽带解析只从分配起点或 CRC24 给出的目录终点推出外层边界和子记录坐标。用 E133 定位锚、后续填充、H2 资源范围及 Qin 非重叠留出核对。没有新增随机划分。

第二项将两个 MCS 66 目录中的五字节描述项，与后续 MCS 的外层头分别读出。按 CRC 通过的 H2 资源顺序核对长度，保留反序、错一位序及未覆盖分配。直接覆盖使用 E130 归档的 LEO 频窗。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_outer_record_lengths --run-id <新编号>
python tools/run_experiment.py audit_outer_directory_links --run-id <另一个新编号>
```

第二项读取已归档的 E134 首跑。四个运行的清单提供输入哈希与对应报告。

[报告](../../reports/E134/2026-11-21-外层打包长度与跨MCS目录.md) · [长度首跑](../../runs/v132-outer-record-lengths/manifest.json) · [长度复跑](../../runs/v132-outer-record-lengths-confirm/manifest.json) · [目录首跑](../../runs/v132-cross-mcs-directory/manifest.json) · [目录复跑](../../runs/v132-cross-mcs-directory-confirm/manifest.json)
