# E047 · 多录波解码链审计

对原 WAV、两份 NPZ 和两份宽带录波的实测投影使用同一时延相关。对 28 个由 CRC 确认的宽带双头部，逐频窗计算 H1、H2 与首个 Code 5 字的理想观测秩。

- [实验计划](../../docs/plans/2026-11-21-cross-capture-decoding-audit.md)
- [解码链证据总表](../../docs/DECODE_MAP.md)
- [运行清单](../../runs/v045-cross-capture-decoding/manifest.json)

复跑命令 `python tools/run_experiment.py audit_cross_capture_decoding --run-id <新编号>`。
