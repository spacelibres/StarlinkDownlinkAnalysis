# E122 · 跨调制的 MCS 校验转移

假设 MCS 66 与 85 共享一组信道校验，MCS 70 与 88 共享另一组。候选配对来自 E070 实测系统区终点。沿用 E083、E085、E120 冻结校验及 E030 的交叉 32QAM 标号。校验、位顺序和评分均无目标组再拟合。

从冻结的 Qin HDF5 频域缓存重新解调十种 16,384 位候选格式。每个分配取末码字，另在多码字分配取第一个非末字。主判据用非末码字，末字只作补充对照。逐码字保留实收位和擦除，逐档案组等权计分，采用 E077 分组和种子 `2026112177`。另用 E030 冻结接收输出及其帧分组检查 250 MS/s 原录波。错矩阵、系统项块内错一位及串行坐标错一位均列为对照。串行错位的区分只用会跨过 512 位边界的行，双方共用实收掩码。

按 LEO 原工程已存的两个 2.5 MS/s 短窗档案、7.5 MS/s 导频规格和 25 MS/s 录波元数据，遍历一槽内全部 1,004 个码字起点。仅按频窗统计每条校验实收项的理想上限。投影使用 E083/E120 的各两族原矩阵，非 E085 新增的三族。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_cross_mcs_checks --run-id v120-cross-mcs-checks
python tools/run_experiment.py audit_cross_mcs_boundary --run-id <新编号>
```

校验重放可用新编号。第二个脚本读取 `v120-cross-mcs-checks` 的冻结证据，不重解调或再选码字。两步记录输入 SHA-256，Qin 本地范围缓存的实际页摘要，以及 LEO 原文件指纹。

[报告](../../reports/E122/2026-11-21-MCS跨调制共享校验.md) · [主运行](../../runs/v120-cross-mcs-checks/manifest.json) · [复跑](../../runs/v120-cross-mcs-checks-confirm/manifest.json) · [边界复验](../../runs/v120-cross-mcs-boundary/manifest.json) · [边界复跑](../../runs/v120-cross-mcs-boundary-confirm/manifest.json)
