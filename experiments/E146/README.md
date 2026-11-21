# E146 · Code 28 软符号与四角判决

沿用 E145 的记录位置和 E021 的生成器。硬判分支查找覆盖整 OFDM 时隙的四角星座，比较整符号擦除与仅保留象限约束。软符号分支在已缓存的三个档案块中检验两种增益与三个固定擦除比例，随后的见证脚本逐块隐藏可判校验位。所有输入只读，保留首次扫描的完整失败分支。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code28_corner_constraints --run-id <新编号>
python tools/run_experiment.py verify_code28_soft_witnesses --run-id <另一新编号>
```

[实验报告](../../reports/E146/2026-11-21-Code28软符号擦除与校验预测.md) · [扫描](../../runs/v144-code28-corner/manifest.json) · [扫描复跑](../../runs/v144-code28-corner-confirm/manifest.json) · [见证](../../runs/v144-code28-soft-witness/manifest.json) · [见证复跑](../../runs/v144-code28-soft-witness-confirm/manifest.json)
