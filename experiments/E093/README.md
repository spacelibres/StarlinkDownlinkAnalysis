# E093 · LEO 上边缘重复位假设与 Code 5

固定 LEO 原工程给出的第二个 OFDM 符号上边缘载波 480 与 486，检验同号关系能否跨目录布局和宽带录波保持。读取外部报告、分析脚本和哈希清单，核对本机能用于独立接收机复验的材料。

运行 `python tools/run_experiment.py audit_leo_copy_transfer --run-id <新编号>`。外部工程默认位于 `../leo-tracker-reduxredux-main`，也可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 均设为 1。

[报告](../../reports/E093/2026-11-21-LEO上边缘重复位与Code5.md) · [主运行](../../runs/v091-leo-copy-audit/manifest.json) · [复跑](../../runs/v091-leo-copy-replay/manifest.json)
