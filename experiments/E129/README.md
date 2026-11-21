# E129 · LEO DS9 双路早期符号与下边缘三位关系

从 LEO Tracker 的 `lower_edge_parity.json` 固定三处符号、FFT 载波和异或目标。从本机两份 DS9 判决 JSON 逐符号复核 RX0、RX1、原门槛及擦除。复用 E087 的资源坐标，比较 Qin 硬判档案和独立 480 MS/s 录波中经过重编码校验的前两个 Code 5 字。原关系来自 Qin 前 78 帧，后续档案仍属同源。DS9 各访次保留原发布评估帧。固定假设逐帧检验，无新增划分、调参或随机种子。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_lower_edge_relation --run-id v127-leo-lower-edge
```

默认只读 `../leo-tracker-reduxredux-main`，可用 `STARLINK_EXTERNAL_LEO_ROOT` 指向同一快照。清单记录本机文件及原始样本期望哈希，逐帧 JSON 保留所有反例与 `?`。已有文件名和运行编号均不覆盖。

[结果与局限](../../reports/E129/2026-11-21-LEO下边缘三位关系.md) · [首跑](../../runs/v127-leo-lower-edge/manifest.json) · [复跑](../../runs/v127-leo-lower-edge-confirm/manifest.json)
