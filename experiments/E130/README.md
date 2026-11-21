# E130 · LEO 频窗对 MCS 66／85 完整码的可辨识性

用 E128 的完整矩阵检验 E122 留下的问题。给定码字位置、星座标号及无噪声带内位，把带外位作为擦除，从完整矩阵直接评估所有线性组合。频窗来自原始 IQ 选段、导频规格、10 MS/s 判决、25 MS/s 目录和 0181 的频偏回执。最宽条件采用所有频窗的载波并集，逐调制枚举全部 1,004 种槽内码字起点。归档不可观测列的两组不相交满秩基作为证据。

沿用 E030 和 E088 的冻结填充区间与留出帧，计算七个末字的剩余自由度。对自由度最低的末字，再以相同观测窗进行硬判消元、软判译码和无噪声对照。扩至系统区末端的 PN15 是单例条件检验。评估样本继承自原宽带捕获，两个运行都是确定性复算。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_full_code_observability --run-id <新编号>
python tools/run_experiment.py replay_leo_padded_code --run-id <另一个新编号>
```

外部目录默认 `../leo-tracker-reduxredux-main`，可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定。第二个脚本读取已归档首跑的几何结果。两组命令各在不同目录复算一次。

[结果](../../reports/E130/2026-11-21-LEO窄带全码可辨识性.md) · [几何](../../runs/v128-leo-full-code-observability/manifest.json) · [几何复跑](../../runs/v128-leo-full-code-observability-confirm/manifest.json) · [实测掩码重放](../../runs/v128-leo-padded-code/manifest.json) · [重放复跑](../../runs/v128-leo-padded-code-confirm/manifest.json)
