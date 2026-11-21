# E137 · Code 5 首记录的 53 帧倒计数

从 E091 固定的目录长度和首记录跨度定位 69 字节共同尾部，用 E078 的结构确认长记录多出的四段 7 字节条目，并核对 E136 的载波位置在两种布局下落在何处。在 E091 的两字证据中读出 16 位候选字段，保留 Code 5、CRC24 核验和逐位实收掩码。[Liu 等的 PDU、Code 5 与宽带数据](https://arxiv.org/abs/2610.29387v2) · [Qin 等的档案](https://www.nature.com/articles/s44459-026-00075-6) · [Morgan 等的帧时钟](https://arxiv.org/html/2611.13655v1#S4)

沿用 E077 随机种子 `2026112177` 的整组划分。在有实收帧计数和计时字且计时字非零的校准帧中，用 $N+K$ 的模值集中度比较 45–61 帧周期。把计数达到最大值的重载帧单独核对相位，冻结后测试另一组及实录。独立 480 MS/s 只用第一帧给定本地相位，测试剩余连续帧。按 E079 的实际选段与名义频窗计算给定布局时的直接覆盖和 Code 5 理想观测秩。无需新增 RF 数据或下载。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_record_countdown --run-id <新编号>
```

[完整结果](../../reports/E137/2026-11-21-Code5首记录倒计数.md) · [正式运行](../../runs/v135-code5-record-countdown/manifest.json) · [复跑](../../runs/v135-code5-record-countdown-confirm/manifest.json)
