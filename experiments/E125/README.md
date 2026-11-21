# E125 · 交叉 32QAM 部分软译码与 MCS 85 计数

使用 E122 已验证的跨调制关系，把 E085 的五族校验用于 MCS 85，把 E120 的两族用于 MCS 88。保留最邻近星座点、仅信道 LLR、正确校验和系统项错位一位的对照。已知 PN15 和记录模板只用于校准或评分，译码仅使用信道 LLR 与冻结矩阵。

- Qin 使用 E077 整档案组划分，排除 E124 与原录波绑定的组。PN15 末字由 E081 的冻结标记确定，从估计起点再留 128 位保护。用训练组已知位估计各标号通道的可靠性，擦除输入为零。
- 250 MS/s 原始 IQ 的 E030 符号使用 32 点边际似然，复噪声方差固定为 0.03。E030 原分组四帧用于选择 0、1、3、6 轮与方法，五帧留出。MCS 88 唯一原录波码字继承 MCS 85 的原录波所选设置。
- 检查 E052 留出记录的 62 个未参与定位的模板位，并在新结果扫描中复验冻结的后继候选位置。E053 已定义的同标签配对不随解码修改。检查一个实际发现的低字节进位，对后一个计数的高字节置零八个 LLR 后重新译码。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_cross32_partial_decode --run-id v123-cross32-partial-decode
python tools/run_experiment.py audit_cross32_decoded_counters --run-id <新编号>
```

[报告](../../reports/E125/2026-11-21-交叉32QAM软译码与计数进位.md) · [译码](../../runs/v123-cross32-partial-decode/manifest.json) · [复跑](../../runs/v123-cross32-partial-decode-confirm/manifest.json) · [计数与屏蔽](../../runs/v123-cross32-decoded-counters/manifest.json) · [复跑](../../runs/v123-cross32-decoded-counters-confirm/manifest.json)
