# E104 · MCS 6 目录描述项的跨度验证

重建 E103 的 CRC24 目录并沿用其目录擦除。按 E036 外层 H2 数量读取完整首个 MCS 6 分配，每个码字仅保留前 10,240 个候选系统位，并按冻结方案逐字去扰。用目录开头的标签顺序选一条或两条 5 字节描述项，连续累加其中的长度读出记录范围。运行程序在查看实收载荷前已确定预测，不从 PN 停止点反推记录数。

分别与独立于目录的 E036 末字 PN 起点、实收零填充、记录起点前缀及同记录长度字段比较。保留固定一条或固定两条描述项的对照、错一或两字节的前缀比较和擦除数。整组校准与留出标签继承 E036，发现阶段看过这份档案。LEO 部分在已知真实资源坐标下只计算直接观测。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs6_body_spans --run-id <新编号>
```

[报告](../../reports/E104/2026-11-21-MCS6记录跨度与填充.md) · [运行](../../runs/v102-mcs6-spans/manifest.json) · [复跑](../../runs/v102-mcs6-spans-confirm/manifest.json)
