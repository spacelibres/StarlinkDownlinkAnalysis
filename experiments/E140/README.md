# E140 · 用先发 `03` 逐位预测后续 `01`

复用 E101 的记录提取和长度表，在 E036 到达时间上校验同计数配对。用 E139 的半字节位序核对对应日程位；保持同一捕获的时轴差异。沿用 E138 的整个 53 帧周期划分和种子 `20261121138`。从比较组同时实收位的 `03`／`01` 差异统计周期，冻结读取规则，然后用留出组核对目标整记录及双方实收位。恢复的位单独计入分母。保留错计数、相邻周期候选和提前记录缺席的档案边界。[E101](../E101/README.md) · [E138](../E138/README.md) · [E139](../E139/README.md)

输入包括 E091 两字结果、E100/E101 已登记的 106 个本机 Qin 页和两份原始宽带的确认记录。运行离线读取已封存页，核对范围及 SHA-256。报告将 Qin 及同源 250 MS/s 与独立 480 MS/s 分开，并列出这两份原 IQ 上的记录格式差异。[Qin 档案来源](https://www.nature.com/articles/s44459-026-00075-6)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_preview_records --run-id <新编号>
```

[完整结果](../../reports/E140/2026-11-21-Code5前置记录与60位子结构.md) · [正式运行](../../runs/v138-code5-preview-records-masks/manifest.json) · [复跑](../../runs/v138-code5-preview-records-confirm/manifest.json)
