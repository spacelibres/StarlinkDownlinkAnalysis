# E128 · MCS 66／85 完整信道校验矩阵与重编码

用 E127 的六族和 E083 校准末字恢复剩余两族。搜索固定低奇偶块和三个高奇偶块的组合，允许第一个系统块有一项实收的活动前缀。用相同旋转的不同校准字作差，剔除共同 PN15 分量。56 个旋转窗、种子 `20261121128`，每窗 24 次差分。已知两族同时用作正对照。余下系统项从填充起点变化拟合。固定 PN15 符号下至多两项的补偿候选，只用非末字校准组选择。最终用 E127 同源组排除后的中间字和 E028/E030 保留的原录波帧评分。

独立实现 GF(2) 奇偶区求逆，按系统位重编码。原录波以固定 6、18、36 轮为检查点，取首次全校验通过。第二条接收路径在每字相同的奇偶区相对坐标屏蔽约 1/16 的完整调制符号，种子 `20261121129`。用未进入译码的复数样本对比重调制及错位奇偶区。交叉 32QAM 最后一个符号含额外标号位，重调制评分省略整个末符号。

记录对照要求原录波正常路径与屏蔽路径逐位一致，再按 E052 的固定区间回放 E053/E126 的字段和协议对照。候选字及全校验是否通过各自归档。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py recover_mcs66_full_code --run-id v126-mcs66-full-code
python tools/run_experiment.py validate_mcs66_full_code --run-id v126-mcs66-full-validation
python tools/run_experiment.py audit_full_mcs85_records --run-id <新编号>
```

[报告](../../reports/E128/2026-11-21-MCS66与85完整信道码.md) · [恢复](../../runs/v126-mcs66-full-code/manifest.json) · [恢复复跑](../../runs/v126-mcs66-full-code-confirm/manifest.json) · [重编码](../../runs/v126-mcs66-full-validation/manifest.json) · [重编码复跑](../../runs/v126-mcs66-full-validation-confirm/manifest.json) · [记录](../../runs/v126-mcs85-full-records/manifest.json) · [记录复跑](../../runs/v126-mcs85-full-records-confirm/manifest.json)
