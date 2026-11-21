# E119 · 弱帧内的资源级尾段边界

只使用 E118 已冻结的相位与粗起点。需要两个导频校准和两个互换频组的测量，都在粗起点前后给出正向留出差，并且起点跨度不超过两槽。对这些帧，另选粗边界之后、末端估相区之前的完整槽估计每载波增益。用常数相位和一次相位漂移两种模型，候选波形从 `R` 开始加载 E025 序列。所有 `R` 都与已测相位模 60 同余。

1. 每槽先按连续非导频资源分块。各块独立移除观测和预测中的常量。整块分给两个互斥频组。每组根据残差平方差选 `R`，再在另一组评价该候选。不得使用候选 H2 或分配长度确定 `R`。
2. 保存两导频校准、两相位响应和两个频组的所有边界。候选均位于搜索区间内部、彼此跨度至多 240 个资源时，才把外扩一个周期的区间作为 H2 限制。这是候选范围，精确读数单独计数。以冻结的 E110 三个旧头部检查覆盖，以保留真实干扰的尾段残差和已知起点注入检验误差。
3. 相位及边界归档两次一致后，比对 E110 的完整分配和补一项末分配的候选。保留相同预算的错边界对照。只依据非 CRC 样点排名，CRC 及既有分配只用于事后评分。

对八个分支一致选中内部单点的帧，追加两描述符搜索。允许两项分配均没有已标记的 PN15 停止，逐个枚举 E067 的宽度和正整数数量，要求恰好闭合到该单点。以同一相位的 ±60、±120、±600 资源作错终点对照。扩展搜索与主边界和区间比较分开归档，CRC 不参与排名。

新增通过 CRC 的候选只在两次头部结果冻结后复核。对比分歧标志位在前 48 资源与其后的卷积输出，另从原始 IQ 取每个预测码字的 128 资源已知 PN15 停止模板。连续五个码字成组并随机互补划分。冻结的模板、相位和码字宽度同预算用于错位置及同宽标签对照。全程保留第 11 时隙的旧头部作为正对照。

边界的连续频块长 32 资源，每相邻两块随机互换分组，种子 `20261121123`。体区验证种子 `20261121124`。合成对照从同一帧较早的实收资源取背景，使用独立尾段锚估计的每载波增益和冻结相位，在候选区间中间一半随机选已知起点。角度和幅度拟合条件与实测不同，按频域合成注入单列。首轮读取 E043 中未同步记录的 `sync` 失败，正式运行按已同步记录取输入。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_weak_tail_endpoint --run-id <新编号>
python tools/run_experiment.py compare_weak_tail_endpoint --run-id <新编号>
python tools/run_experiment.py decode_two_tail_allocations --run-id <新编号>
python tools/run_experiment.py verify_two_tail_allocations --run-id <新编号>
```

[报告](../../reports/E119/2026-11-21-弱帧资源级边界和第14时隙H2.md) · [初次失败](../../runs/v117-ut-weak-resource-end/manifest.json) · [边界](../../runs/v117-ut-weak-resource-end-final/manifest.json) · [边界复跑](../../runs/v117-ut-weak-resource-end-final-confirm/manifest.json) · [区间约束](../../runs/v117-ut-weak-resource-h2/manifest.json) · [区间复跑](../../runs/v117-ut-weak-resource-h2-confirm/manifest.json) · [两项分配](../../runs/v117-ut-weak-two-allocations/manifest.json) · [两项复跑](../../runs/v117-ut-weak-two-allocations-confirm/manifest.json) · [体区](../../runs/v117-ut-weak-two-body/manifest.json) · [体区复跑](../../runs/v117-ut-weak-two-body-confirm/manifest.json)
