# E133 · 高阶 MCS 的跨格式记录、计数与外层长度

沿用 E128、E131、E132 的宽带判决，确认整个分配的系统位齐全。按固定 PN15 约定逐字解扰并按时间顺序拼接。在每个分配的字节流中以七个固定字节寻找 20 字节候选。按同一分配内其余 18 个头字节分组，比较候选两字节字段大端与小端的模差。对 E122 的 Qin 硬判留出，排除组 19，要求全部定位字节实收，同掩码复验正确 PN15 和错一位相位。档案中缺少相邻码字时保留原字边界。

外层长度沿用 E104 的已验证字段规则，从每个完整原录波分配的零偏移开始走已支持的头标记。固定预测终点后才检查下一类型或填充，并比较四个错位。完整性实验限定外层预测范围，检验 CRC16、CRC32、CRC32C 与已恢复的本地目录 CRC24。五种算法、两个覆盖起点和两种尾字节序都归档。等长记录之间也比较残差，保留直接匹配及同余常量检验的结果。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_cross_mcs_record_fields --run-id <新编号>
python tools/run_experiment.py audit_cross_mcs_integrity_fields --run-id <另一个新编号>
```

第二项读取已归档首跑的区间。原录波锚为回顾性发现，Qin 保留组用于固定条件验证。没有新增随机划分或调参。

[报告](../../reports/E133/2026-11-21-跨MCS记录长度与计数.md) · [字段首跑](../../runs/v131-cross-mcs-records/manifest.json) · [字段复跑](../../runs/v131-cross-mcs-records-confirm/manifest.json) · [完整性首跑](../../runs/v131-cross-mcs-integrity/manifest.json) · [完整性复跑](../../runs/v131-cross-mcs-integrity-confirm/manifest.json)
