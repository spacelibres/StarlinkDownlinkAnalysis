# E136 · LEO 跨码字奇偶与 Code 5 记录跨度

从 E135 冻结 `shared-28`、`shared-29`、`shared-48`、`shared-53` 的坐标和极性。合并 E078、E080 和 E091 的实收掩码与前两个 Code 5 字，核对外层码与目录 CRC24。使用相同两份原始宽带的校验通过数据对照。Qin 档案的原实轴符号用早于目标位置的已知 H1 前 48 编码位校定，统一翻转仅影响奇数项异或的绝对极性。[LEO 原工程假设](https://github.com/misko/leo-tracker-reduxredux/blob/main/reports/2026_09_28_sequence_semantics/HEADER_BLOCK_RANK.md) · [Qin 档案来源](https://www.nature.com/articles/s44459-026-00075-6) · [宽带分配及信道码参考](https://arxiv.org/abs/2610.29387v2)

已用过的 Qin 0–77 帧和 E135 的 104 个后续帧排除在新增检验外，余 827 帧。同一捕获内沿用 E077 的固定整组划分与种子 `2026112177`，保留其中 502 帧拟合条件多数值、325 帧留出。每个布局分别由目录首字节和实收跨度字段确定。比较只按目录长度分组与按目录长度和首记录跨度共同分组，平局或训练缺失时按擦除记。文内和产物中的帧数均不视为独立采集次数。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_layout_relation --run-id <新编号>
```

[结果与证据边界](../../reports/E136/2026-11-21-LEO跨码字奇偶与记录跨度.md) · [正式运行](../../runs/v134-leo-layout-polarity/manifest.json) · [确定性复跑](../../runs/v134-leo-layout-confirm/manifest.json)
