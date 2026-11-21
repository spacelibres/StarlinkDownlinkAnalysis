# E105 · MCS 6 目录对后续分配的长度约束

重新校验 E103 的目录，只移除 E104 已归属到 MCS 6 自身的首段描述项。将其余描述项按原顺序分配给经外层 CRC 验证的 H2 条目。使用每码字系统区宽度与码字数量给出的容量区间，逐个列举连续记录的组合，保留多解、后续失败及能完整耗尽列表的情形。后续数据仅在预测冻结后评分。

在固定的 HDF5 页缓存中按 E081 的复增益与掩码重收 101 个后续分配。用 E081 已固定的 PN 质量门槛选择端点见证。原顺序、逆序与 199 次帧内置换使用同一规则，每帧包含相同的一组长度值。种子固定为 `20261122105`，继承 E104 的分组。另一条验证支路在候选记录起点读取四字节头及其独立长度字段，对照前后错位。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs6_cross_allocation --run-id <新编号>
```

[报告](../../reports/E105/2026-11-21-MCS6目录跨分配验证.md) · [运行](../../runs/v103-cross-allocation/manifest.json) · [复跑](../../runs/v103-cross-allocation-confirm/manifest.json)
