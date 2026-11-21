# E058 · 卷积头部的留出输出验证

轮流屏蔽率 1/3 卷积码的一路输出，用另外两路选择极性和译码，最后用留出输出与 CRC 检验。比较 Qin 公开档案、250 MS/s 偏移时轴与当前窄带的观测范围。

- [实验设计](../../docs/plans/2026-11-21-header-parity-lanes.md) · [报告](../../reports/E058/2026-11-21-卷积头部第三路验证.md)
- [运行清单](../../runs/v056-header-parity-lanes/manifest.json) · [复跑](../../runs/v056-header-parity-confirm/manifest.json)

`python tools/run_experiment.py audit_header_parity_lanes --run-id <新编号>`。使用本地已归档频域符号与 IQ 派生观测，所有输入哈希随结果保存。
