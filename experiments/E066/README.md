# E066 · 无参考估计 LEO 的 60 位周期形状

按配对的连续帧组划分，从 LEO 原始 IQ 的校准段学习 60 位循环形状。宽带参考只由另一个程序读取，用来比较已经冻结的盲估计。

- [实验设计](../../docs/plans/2026-11-21-leo-blind-word.md) · [实验报告](../../reports/E066/2026-11-21-LEO盲估计与位擦除.md)
- [盲估计与擦除](../../runs/v064-leo-blind-consensus/manifest.json) · [盲估计复跑](../../runs/v064-leo-blind-consensus-confirm/manifest.json)
- [宽带比对](../../runs/v064-leo-blind-consensus-compare/manifest.json) · [比对复跑](../../runs/v064-leo-blind-consensus-compare-confirm/manifest.json)

先用 `py -3.14 tools/run_experiment.py audit_leo_blind_word --run-id <新编号>` 测量，再用 `py -3.14 tools/run_experiment.py compare_leo_blind_word --run-id <新编号>` 复验归档的盲估计。两次估计和随后比对的来源见运行清单。外部目录可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定。
