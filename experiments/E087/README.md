# E087 · LEO 早期符号的资源归类

按物理载波和 OFDM 符号编号重定位 LEO 原工程导出的 24 个符号。用同源的宽带原始硬判、冻结 Code 5 生成矩阵和独立解调前端的参考变换逐位复核。另用 Qin 档案核查可见坐标及 DS7 字样的重复性。

运行 `python tools/run_experiment.py audit_leo_early_symbol_roles --run-id <新编号>`。默认从 `../leo-tracker-reduxredux-main` 只读。可设 `STARLINK_EXTERNAL_LEO_ROOT` 指向相同输入。

[报告](../../reports/E087/2026-11-21-LEO早期符号与Code5坐标.md) · [运行清单](../../runs/v084-leo-early-symbols/manifest.json) · [复跑清单](../../runs/v084-leo-early-symbols-confirm/manifest.json)
