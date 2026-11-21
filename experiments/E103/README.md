# E103 · 从 Code 5 向 MCS 6 转移目录校验

冻结 E091 的 CRC24 多项式、初值、余数和字节位序。输入为 E036 已登记的 Qin BPSK 数据与同档案其它 MCS 切片。只用 E038 前置 Code 5 已估出的相位校正 `+511`，不根据 MCS 6 的 CRC 选相位。不确定的坐标沿用 E039 的可靠度掩码。区分原始硬判、相位校正的硬判与 CRC 唯一补全的擦除，记录逐字结果。

沿用 E036 整个档案区块的校准和留出标签。直接检验错 PN、错位序、同长字节偏移和其它 MCS。对通过的目录，检查旧 C/E 描述项在目录长度和 CRC 之前是否封闭，并投影至原 LEO 工程的频窗。登记 MCS 70 单点三位关系的旋转负结果，候选坐标来自探索扫描。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_mcs6_directory_crc --run-id <新编号>
```

[报告](../../reports/E103/2026-11-21-MCS6跨格式目录校验.md) · [运行](../../runs/v101-mcs6-directory/manifest.json) · [复跑](../../runs/v101-mcs6-directory-confirm/manifest.json)
