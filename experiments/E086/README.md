# E086 · LEO DS7 的 60 位重复字跨录波验证

用冻结的 E025 宽带模板预测 LEO 原工程的 DS7 字族，逐项核对双接收机导出、原预留组、全局极性和偶校验。输入详情和归因限制见[实验报告](../../reports/E086/2026-11-21-LEO-DS7重复字与宽带参考.md)。

运行 `python tools/run_experiment.py audit_leo_ds7_reference_transfer --run-id <新编号>`。默认从 `../leo-tracker-reduxredux-main` 只读，也可用 `STARLINK_EXTERNAL_LEO_ROOT` 指向另一份相同输入。

- [单符号方向首轮](../../runs/v083-leo-ds7-reference/manifest.json)与[复跑](../../runs/v083-leo-ds7-reference-confirm/manifest.json)
- [双符号方向运行](../../runs/v083-leo-ds7-family/manifest.json)与[复跑](../../runs/v083-leo-ds7-family-confirm/manifest.json)
