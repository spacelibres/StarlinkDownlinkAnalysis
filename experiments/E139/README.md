# E139 · Code 5 时隙半字节的组内帧偏移

以 E138 的实收七字节及相同的 53 帧分组为输入。把每个半字节的四位分别映射到 $N+4i+d$，$d$ 取 0–3。比较同一组内的所有偏移，目标必须同时满足独立帧计数及与源帧相同的到达时轴。保留整组落在周期内、整组落在周期后及跨越边界的标记。剔除提示该假设的 Qin 第 42 帧所属周期后比较全部 24 种比特到偏移的置换。[E137](../E137/README.md) · [E138](../E138/README.md) · [Qin 原始档案的选择范围](https://www.nature.com/articles/s44459-026-00075-6)

沿用 E138 的整组划分与随机种子 `20261121138`，本实验未另行拟合或重抽。用旧 480 MS/s 原始宽带的 E010/E138 已验证帧比较未置位位置。两次复跑使用同一批实收数据。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_slot_bitmap --run-id <新编号>
```

[结果与证据边界](../../reports/E139/2026-11-21-Code5时隙组内帧偏移.md) · [正式运行](../../runs/v137-code5-slot-bitmap/manifest.json) · [确定性复跑](../../runs/v137-code5-slot-bitmap-confirm/manifest.json)
