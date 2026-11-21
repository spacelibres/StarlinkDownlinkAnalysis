# E061 · 槽内调制边界与档案硬判

对照 Qin 硬判档案与已配对的 250 MS/s 原 IQ，检验同一个 OFDM 槽里的 16QAM 到交叉 32QAM 分配切换。输出包括 19 个格式的覆盖清单、逐边界星座证据、错位对照和截断记录。

- [实验设计](../../docs/plans/2026-11-21-qin-mixed-slot.md) · [报告](../../reports/E061/2026-11-21-槽内调制与硬判档案.md)
- [117 帧运行](../../runs/v059-qin-mixed-final/manifest.json) · [全部档案帧](../../runs/v059-qin-mixed-stable/manifest.json) · [复跑](../../runs/v059-qin-mixed-verify/manifest.json)
- [检测到原 IQ 截断之前的首次失败](../../runs/v059-qin-mixed-slot/manifest.json)

`python tools/run_experiment.py audit_qin_mixed_slot --run-id <新编号>` 使用已随仓库归档的 117 帧。加 `--full-hard-cache` 从 `.cache/qin-mat-pages` 的既有范围缓存复查全档案。扩展结果保存实际读取的区间及哈希。
