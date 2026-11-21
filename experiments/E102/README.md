# E102 · 在标称 25 MS/s 频窗里搜索 Code 5 后续记录帧号

验证 E101 后续记录的 14 位帧计数能否从真正的软件限带波形读到。输入为 480、250 MS/s 两份封存的原始 IQ 与公开同步模板。读取 LEO Tracker 两份带宽配置及其登记指纹，从同一份录波生成上下边缘两种复数信号。用限带波形独立定位 PSS、SSS，偶数导频用于恢复相位、奇数导频用于计算校验 EVM。

接收端给定首 Code 5 字的资源号、系统字的解扰及 E101 记录格式。第 2 至第 8 个码字内逐字节扫描五个可观察字节，记录 `01` 类型、计数范围及每个候选位置。相邻已接纳机会每两项一组，以种子 `2026112202` 置换整组，检测一个跨组相容的计数起点。时间反向和 BPSK 极性反转使用相同的选组与规则。宽带目录、记录起点和真实帧计数只用于评分。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_narrow_record_search --run-id <新编号>
```

[报告](../../reports/E102/2026-11-21-25MSps限带后续记录帧号.md) · [初轮](../../runs/v100-narrow-record-search/manifest.json) · [最终运行](../../runs/v100-narrow-record-final/manifest.json) · [复跑](../../runs/v100-narrow-record-confirm/manifest.json)
