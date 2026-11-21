# E126 · MCS 85 的独立记录边界

用 E125 的 250 MS/s 软判和 E052 的四个校准帧学习 64 字节逐位模板。学习阶段用既有 20 字节定位结果，验证阶段只读取首四字节，完整实收且最多相差两位时登记。偏移 4–7 已有可变字段，跳过这段。固定规则寻找从偏移 8 起逐字节一致率至少 0.8，随后四字节一致率低于 0.7 的首个位置。

在 E052 五帧留出和排除同源组的 E077 Qin 档案留出中，分别计算未参与定位的字节一致率、错 PN15、偏移对照及冻结标签的后继位置。档案用 E124 的分配坐标，扫描整段系统字节，不读取 E124 已检测的记录位置。原录波用同一批 E052 已知后继间距复验 IPv4、IPv6 和 Ethernet FCS。LEO 投影只使用原工程频窗和宽带坐标。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs85_record_boundary --run-id <新编号>
```

[实验报告](../../reports/E126/2026-11-21-MCS85记录的20字节边界.md) · [运行](../../runs/v124-mcs85-record-boundary/manifest.json) · [复跑](../../runs/v124-mcs85-record-boundary-confirm/manifest.json)
