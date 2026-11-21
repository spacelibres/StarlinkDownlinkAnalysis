# E065 · 用冻结 60 位模板预测 LEO 原始 IQ

使用 E025 宽带得到的固定字，在 E064 校准块选每帧相位和复增益。将非导频预测直接与原时域采样对比，并检验双路各自可见载波的相位转移。

- [实验设计](../../docs/plans/2026-11-21-leo-tcode-iq-prediction.md) · [报告](../../reports/E065/2026-11-21-LEO非导频模板与双路转移.md)
- [首轮](../../runs/v063-leo-tcode-iq/manifest.json) · [首轮复跑](../../runs/v063-leo-tcode-iq-confirm/manifest.json)
- [双路转移](../../runs/v063-leo-tcode-pair-transfer/manifest.json) · [复跑](../../runs/v063-leo-tcode-pair-confirm/manifest.json) · [逐帧数值](../../runs/v063-leo-tcode-pair-transfer/artifacts/experiments/E065/results/leo_tcode_iq_prediction.json)

`py -3.14 tools/run_experiment.py audit_leo_tcode_iq_prediction --run-id <新编号>`。读取原工程目录及 E025、E056、E064 的归档回执。可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定外部目录。
