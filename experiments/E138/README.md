# E138 · Code 5 尾部的未来帧时隙状态

从 E137 的 69 字节对齐尾区取零基字节 36–42，用 E091 实收掩码检验每个半字节。用 E137 已冻结的 53 帧周期及相位限制当前周期边界。比较低半字节在先、间距 4 帧，与反序、间距 3／5／8 帧和整体 ±1 帧。目标须有独立时间索引和已核验的帧计数，时间容差为 0.05 帧。档案缺席仅记为未出现匹配记录。[E137](../E137/README.md) · [Qin 的测时与波束讨论](https://arxiv.org/abs/2501.05302) · [公开频域档案](https://www.nature.com/articles/s44459-026-00075-6)

首尾各一个周期未进入比较。剩余 107 个整组用随机种子 `20261121138` 划分，72 组用于比较，35 组留出。同一次采集的相邻周期仍可能相关。去重统计使用目标计数与周期，并保留仅有异时轴匹配的记录。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_future_slots --run-id <新编号>
```

[结果及证据边界](../../reports/E138/2026-11-21-Code5未来帧时隙状态.md) · [正式运行](../../runs/v136-code5-future-slots-final/manifest.json) · [复跑](../../runs/v136-code5-future-slots-confirm/manifest.json)
