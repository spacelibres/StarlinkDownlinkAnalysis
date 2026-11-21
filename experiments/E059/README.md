# E059 · 双时轴循环前缀差分

在 250 MS/s 原 IQ 上利用主时轴循环前缀与符号尾部的复制关系做差。比较三类时隙与错间距对照，并用完整 FFT 的训练块检验偏移时轴导频能否留在差分观测中。

- [实验设计](../../docs/plans/2026-11-21-overlap-prefix-cancellation.md) · [报告](../../reports/E059/2026-11-21-双时轴循环前缀差分.md)
- [短前缀模型](../../runs/v057-overlap-prefix-final/manifest.json) · [FFT 校准](../../runs/v057-overlap-prefix-fft/manifest.json) · [复跑](../../runs/v057-overlap-prefix-confirm/manifest.json)
- [固定窗越界的首轮](../../runs/v057-overlap-prefix/manifest.json)

`python tools/run_experiment.py audit_overlap_prefix_cancellation --run-id <新编号>`
