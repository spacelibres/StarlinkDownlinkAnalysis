# E062 · Code 21 的留出校验与单符号纠正

复用 E034 的硬判和软符号切片，以及旧宽带录波得到的冻结 Code 21 生成器。比较逐槽相位搜索和单个 QPSK 符号的局部变化，在未参与候选选择的校验位上验算。

- [实验设计](../../docs/plans/2026-11-21-code21-quarter-slips.md) · [报告](../../reports/E062/2026-11-21-Code21留出校验与单符号纠正.md)
- [512 位校验块分组](../../runs/v060-code21-quarter-slips/manifest.json) · [32 位分散校验组](../../runs/v060-code21-quarter-stratified/manifest.json)
- [最终运行](../../runs/v060-code21-one-symbol/manifest.json) · [复跑](../../runs/v060-code21-one-symbol-confirm/manifest.json)

`python tools/run_experiment.py audit_code21_quarter_slips --run-id <新编号>`
