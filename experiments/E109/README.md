# E109 · 偏移弱时间轴的第二头部与下游原始 IQ

沿用 E043 至 E045 的两条时轴、28 个弱帧时隙和候选 H2 起点 8420。首轮在两个相位模式下用已知首字节 `0x34`、合法块数及条目数、20 位描述符及零填充约束卷积译码。前缀和块数按第一完整 114 资源的度量确定，随后按选定容量解其余块。终止 trellis 的 CRC 位仍自由，CRC 在得到判决后才计算。两相位分别由已冻结的 H1 及当前 H2 已知首字节得出。[Liu 等头部结构](https://arxiv.org/html/2610.29387v2#S4) · [Viterbi 卷积译码](https://doi.org/10.1109/TIT.1967.1054010)

验证包含轮流屏蔽每一路卷积生成多项式、对前 16 位保护后将其余编码三元组分为八组、±1 与 ±114 资源错位及 63 组保持编码前缀的三元组置乱。种子 `20261121109`。另冻结 E044 的另一套导频校准及独立 480 MS/s 强帧正对照。两次隔离重放后，从单个 H2 候选计算两个最后码字的系统区终点。体区检验读取此前未参与头部译码的远端原始资源。H2 候选在这一步保持冻结。

体区检验由 250 MS/s 原始 CI16 经 E014 重采样至 480 MS/s，使用 E043 弱网格位置、E044 两套采样修正和已知符号首字节定相。采用 E051 的 128 个资源窗口、E028／E030 星座标签、E029 PN15 和冻结的系统区宽度。逐点比较有符号归一化复数内积，记录预言位置及前后各 800 个资源、63 个整 16 资源模板块置乱（种子 `20261121110`）和其他 27 个弱时隙。对 MCS 92 跳过系统位与校验位共用的那一个资源后，再测试随机序列能否延续。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_ut_offset_constrained_h2 --run-id <新编号>
python tools/run_experiment.py verify_ut_offset_h2_body --run-id <新编号>
```

后一个入口核对两份正式头部结果一致后才读原始体区。所有本录波内的时间、拟合、擦除和置乱分支共享原始 IQ。[报告](../../reports/E109/2026-11-21-偏移弱时间轴H2及分配复验.md) · [头部首轮](../../runs/v107-ut-offset-constrained-h2/manifest.json) · [正式头部](../../runs/v107-ut-offset-constrained-h2-final/manifest.json) · [头部复跑](../../runs/v107-ut-offset-constrained-h2-confirm/manifest.json) · [体区首轮](../../runs/v107-ut-offset-h2-body/manifest.json) · [体区首轮复跑](../../runs/v107-ut-offset-h2-body-confirm/manifest.json) · [正式体区](../../runs/v107-ut-offset-h2-body-final/manifest.json) · [体区复跑](../../runs/v107-ut-offset-h2-body-final-confirm/manifest.json)
