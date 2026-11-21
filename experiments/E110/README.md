# E110 · 从弱帧原始体区反推下游分配与 H2

冻结 E043／E044 的弱时轴捕获、两套公开导频拟合和 E045 的 H2 起点 8420。两套拟合保留 E044 的整帧分组和重叠帧。用 28 个弱帧重新生成实收资源格。已知首字节 `0x34` 提供每帧相位。扫描从资源 9218 开始，位于最长七块 H2 的末端。体区扫描程序不读 H2 可变位或 CRC 判决。[匹配滤波方法](https://doi.org/10.1109/TIT.1960.1057571)

在八个已知格式中，用对应星座标号构造 128 个完全系统区 PN15 资源的匹配模板。候选在前窗的有符号归一化复数匹配至少为 0.30。系统区后 64 个资源的同序列延续匹配至多为 0.25，下降至少 0.16。若系统位与校验位共用资源，后窗跳过该资源。候选还必须强于 31 次整 16 资源块置乱的得分。种子为 `20261121111`。门槛由 E109 的同录波结果及探索扫描确定。两种导频拟合都通过的格式和末端才进入下一阶段。两拟合共用一次 IQ。

给定 2 至 7 个 H2 编码块，起点为 `8420 + 114×块数`。从这个起点开始，只有相邻端点差为候选格式码字长度的正整数倍时才连接。保留所有符合描述符容量的路径及两个前缀标志备选。两份体区扫描归档完全一致后，第二程序才读取 H2 软观测。从已知首字节之后到最短候选的 CRC 之前，用同一观测区给全部候选评分。路径确定后，单独从末端 42 个资源的约束卷积状态恢复八位 CRC，包括六步尾随的输出，再与选中头部的信息位核对。并列路径保持未决。概率对照为原三个帧组内的全部非零循环移位，整帧置换冻结的体区路径。[Liu 等的 H2 格式](https://arxiv.org/html/2610.29387v2#S4)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_ut_offset_body_markers --run-id <新编号>
python tools/run_experiment.py decode_ut_offset_body_first --run-id <新编号>
```

[报告](../../reports/E110/2026-11-21-弱帧体区反推头部与CRC.md) · [体区扫描](../../runs/v108-ut-offset-body-markers/manifest.json) · [扫描复跑](../../runs/v108-ut-offset-body-markers-confirm/manifest.json) · [头部解码](../../runs/v108-ut-offset-body-first/manifest.json) · [解码复跑](../../runs/v108-ut-offset-body-first-confirm/manifest.json)
