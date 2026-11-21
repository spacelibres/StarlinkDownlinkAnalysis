# E132 · MCS 71、75、78、81 的共享高地址奇偶校验

将 E128 的八族系统列按系统区右端对齐，移出左端的列作为已知零。保留原奇偶列坐标。按 E127 组划分和组 19 排除规则评分。实收码字比较规则包含正确对齐、系统项错移一位及按左端对齐。任何统计只使用两种候选共同有实收判决的行，缩短列的零已知。

原录波在 250 MS/s 宽带接收器的均衡输出中屏蔽尚待确定的奇偶区。用种子 `20261121132` 另外屏蔽高地址区的 51 个完整符号，重放系统位和已建模的奇偶位。以冻结 PN15 做事后评分。原始接收模型的整帧拆分沿用 E030。另限定绝对块 23 的三项和四项纯奇偶候选，用相同数值的独立初始化种子在 56 个旋转窗检验校准组差分。两个试验均保留结果与对照。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs_lower_rate_core --run-id <新编号>
python tools/run_experiment.py probe_lower_rate_parity --run-id <另一个新编号>
```

[报告](../../reports/E132/2026-11-21-较低码率共享奇偶校验.md) · [矩阵及原录波](../../runs/v130-mcs-lower-rate-core/manifest.json) · [复跑](../../runs/v130-mcs-lower-rate-core-confirm/manifest.json) · [纯奇偶探索](../../runs/v130-lower-parity-probe/manifest.json) · [探索复跑](../../runs/v130-lower-parity-probe-confirm/manifest.json)
