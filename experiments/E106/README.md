# E106 · LEO 上边缘跨导频残余相位

使用 LEO Tracker 的 DS7、DS8、DS9 实收导频矩阵。检验 7.5 MS/s 上边缘在原有普通残余频偏校正后，一个载波子集的逐行相位能否预测另一个子集。复用 E099 的档案校验，保留源文件和输入账本的 SHA-256。

每帧 150 行、8 根导频。原档案用偶数行拟合频偏，本实验也只用这些行估计每根载波的固定复增益。奇数行按载波索引奇偶分成两组，每组四根。各组以固定增益为权重投影到一个复数，双向用预测组的单位相位校正另一组，按目标投影幅值归一化。得分是相对于只使用原固定频偏及增益的同相增量。每帧两个方向等权，窗内各帧等权，汇总各接收机窗等权。

对照包括相位取反、偏移一个留出行（17.6 µs）及 199 组非零循环移位。循环移位保留预测组的整段时序结构。同一采集会话内的访次、接收机、帧及两个转移方向使用同一偏移，跨会话使用种子 `20261121106` 抽取偏移，并在结果中保留分配。行、帧及偏移组均为条件诊断。档案包含三个既有采集会话。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_upper_pilot_residuals --run-id <新编号>
```

[报告](../../reports/E106/2026-11-21-LEO上边缘残余相位.md) · [运行](../../runs/v104-leo-upper-pilot-residuals/manifest.json) · [复跑](../../runs/v104-leo-upper-pilot-residuals-confirm/manifest.json) · [逐窗与逐帧数值](../../runs/v104-leo-upper-pilot-residuals/artifacts/experiments/E106/results/leo_upper_pilot_residuals.json)
