# E063 · 从前段 Code 5 估计后段 Code 21 的边缘相位

固定宽带档案的头部坐标和 Code 21 生成器。用较早的 Code 5 码字估计 +511 响应，再检验同帧较晚的 Code 21。擦除重建作为另一条计算路径，分别保留实测校验和重编码补出的位。

- [实验设计](../../docs/plans/2026-11-21-code5-to-code21-edge.md) · [报告](../../reports/E063/2026-11-21-Code5到Code21的边缘相位.md)
- [首轮](../../runs/v061-code5-to-code21-edge/manifest.json) · [最终运行](../../runs/v061-code5-to-code21-edge-final/manifest.json) · [重复运行](../../runs/v061-code5-to-code21-edge-confirm/manifest.json)
- [完整位串 NPZ](../../runs/v061-code5-to-code21-edge-final/artifacts/experiments/E063/results/code21_edge_reconstructed_words.npz) · [校验位和对照记录](../../runs/v061-code5-to-code21-edge-final/artifacts/experiments/E063/results/code5_to_code21_edge.json)

执行 `python tools/run_experiment.py audit_code5_to_code21_edge --run-id <新编号>`。输入文件指纹由脚本与既有运行清单核对。
