# E070 · PN15 填充的末字分布与字节对齐

用冻结的系统区右端和标号，在全档案扫描每个候选码字。对强填充估计起点，比较不同对齐约束在留出帧的证据。

- [实验计划](../../docs/plans/2026-11-21-pn-padding-starts.md) · [报告](../../reports/E070/2026-11-21-PN填充末字与字节格点.md)
- [档案运行](../../runs/v068-pn-padding-starts/manifest.json) · [档案复跑](../../runs/v068-pn-padding-starts-confirm/manifest.json)
- [软符号对照](../../runs/v068-pn-padding-raw/manifest.json) · [对照复跑](../../runs/v068-pn-padding-raw-confirm/manifest.json)

`python tools/run_experiment.py audit_pn_padding_starts --run-id <新编号>` 读取已保存的 Qin 分块缓存。随后用 `python tools/run_experiment.py verify_pn_padding_grid_raw --run-id <新编号>` 复验归档格点，经 E028/E030 的原录波软符号，无需重新下载。
