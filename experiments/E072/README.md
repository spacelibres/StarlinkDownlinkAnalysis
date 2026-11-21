# E072 · LEO 上边缘投影和帧内相位

把 E014 的 250 MS/s 原始 CI16 变频、限带和抽取到 E071 的 25 MS/s 上边缘几何。比较额外频偏和滤波响应如何影响目标载波 +511，以及导频能否表征最外载波的可用性。

运行 `python tools/run_experiment.py audit_leo_edge_passband --run-id <新编号>`。依赖 250 MS/s 的原文件、E014 的宽带字和时轴、E013 的校验矩阵以及 E071 几何归档。

- [实验设计](../../docs/plans/2026-11-21-leo-edge-passband.md) · [报告](../../reports/E072/2026-11-21-LEO上边缘限带与相位.md)
- [首轮全局相位](../../runs/v070-leo-edge-passband/manifest.json) · [最终运行](../../runs/v070-leo-edge-passband-final/manifest.json) · [复跑](../../runs/v070-leo-edge-passband-confirm/manifest.json)
