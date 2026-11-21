# E118 · 弱偏移网格的独立尾段相位

从 250 MS/s 原始 IQ 重建 E044 的两套偏移网格。先冻结应用解析 PN 的尾段结果，再与 E110 的分配路径和既知 H2 核对。尾段搜索不读取 H2 可变位或 CRC。

1. 在每个完整 OFDM 槽的前 960 个非导频资源中保留 16 组连续 60 资源。按完整块切训练和留出，并互换。减去每块常量，用 E025 冻结的 60 位字与全部循环移位拟合。后部 1、2、4、8、16 槽为预定候选长度。选择长度和相位只用训练数据，留出预测用四个频区各自的训练增益。
2. 两种导频校正和频组互换须选同一个相位。留出误差改善须为正，且胜过同预算的错槽 PN 和 31 个保持组成的置乱字。同录波主时轴上 E025 测得的已知相位为正对照。对缺失最后时隙的录波保留擦除。
3. 相位成功后，用不参与末段估相的早期槽估计可能的尾段开始范围。随后对 E110 的体区路径测试终点的模 60 约束，在可满足时再尝试补一项末端分配。头部候选只依据非 CRC 符号及已冻结的体区结果选择，CRC 单独读出。

分配复核只接收两个冻结相位结果逐项一致的输入。四分支粗起点跨度至多两槽时，取观测首槽范围向两端扩展两槽作为末端候选区间。保留符合范围且与相位同余的已有码字路径。也允许从 H2 末端或已证体区前缀接一条末分配，以 E067 的冻结逐字宽度约束正整数码字数。每个 H2 首部只用最短候选 CRC 之前的共同符号排名。另用前 16 个信息位对应的 48 个编码资源选前缀，然后只比较该前缀下的描述符并评分，作为单独分支。冻结错相位为 ±1、±2、±8、30，应用完全相同搜索和 CRC 门槛。两个导频拟合必须同词并通过 CRC 才保留新候选。

扩大路径后发现第 27 时隙的同宽标签竞争，新增一项只筛选真实体区标记路径的对照，分别记录这两种证据条件。首轮扩大路径结果保留。粗起点跨度衡量拟合一致性，另保存互换频率留出在起点前后的得分差。

尾段置乱种子为 `20261121122`。首轮两份相位数值相同，但 SSS 输入路径使用运行目录下的临时路径，导致逐项比较失败。随后改用固定项目路径登记相同字节的指纹，两次正式结果逐字节一致。原失败记录保留。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_weak_tail_phase --run-id <新编号>
python tools/run_experiment.py audit_weak_tail_paths --run-id <新编号>
```

[报告](../../reports/E118/2026-11-21-弱偏移网格的尾段相位与H2约束.md) · [相位正式运行](../../runs/v116-ut-weak-tail-phase-final/manifest.json) · [相位复跑](../../runs/v116-ut-weak-tail-phase-final-confirm/manifest.json) · [初次路径读取失败](../../runs/v116-ut-weak-tail-paths/manifest.json) · [补末项路径](../../runs/v116-ut-weak-tail-paths-final/manifest.json) · [补末项复跑](../../runs/v116-ut-weak-tail-paths-final-confirm/manifest.json) · [实收标记路径](../../runs/v116-ut-weak-tail-paths-markers/manifest.json) · [标记路径复跑](../../runs/v116-ut-weak-tail-paths-markers-confirm/manifest.json)
