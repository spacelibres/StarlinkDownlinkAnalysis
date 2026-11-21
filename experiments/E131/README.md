# E131 · MCS 70／88 完整等价码与跨码率结构

用 E128 的八族构造跨码率候选。系统区移位 512 位，原奇偶块 24 移至扩展矩阵的虚拟列。将三个涉及该块的行族两两异或，获得两族独立投影。用 E127 的 MCS 70 校准中间字，在新增系统块中比较零项与 512 个单项候选。排除与原波形重叠的归档组 19，用未参与拟合的 MCS 70 和 88 字评分。

实测路径继承整帧拆分和 6、18、36 轮译码检查点。虚拟列输入零 LLR，冻结 PN15 只用于评分。用种子 `20261121131` 屏蔽约 1/16 的完整奇偶区调制符号，比较正常与屏蔽译码。再用投影矩阵独立重编码，对比屏蔽的实际复数符号及错移奇偶区。最后对 E130 的 LEO 频窗并集重做全起点列基检验。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py recover_mcs70_full_code --run-id <新编号>
python tools/run_experiment.py validate_mcs70_full_code --run-id <另一个新编号>
python tools/run_experiment.py audit_mcs70_leo_observability --run-id <另一个新编号>
```

后两条命令读取已归档的首跑矩阵。每项分析都有自己的复跑和输入哈希。仪器数据、导频归档和行业删余实现均保留各自的来源。

[结果](../../reports/E131/2026-11-21-MCS70与88完整信道码.md) · [矩阵](../../runs/v129-mcs70-full-code/manifest.json) · [矩阵复跑](../../runs/v129-mcs70-full-code-confirm/manifest.json) · [原始录波](../../runs/v129-mcs70-full-validation/manifest.json) · [录波复跑](../../runs/v129-mcs70-full-validation-confirm/manifest.json) · [LEO 几何](../../runs/v129-mcs70-leo/manifest.json) · [LEO 复跑](../../runs/v129-mcs70-leo-confirm/manifest.json)
