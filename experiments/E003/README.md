# E003 · 导频与局部编码候选

参考导频与未知载波能否区分；数据是否足够可观测？

输入：两份 selected IQ、本地参考序列，部分使用 WAV 与跨组结果。

方法：参考相关、非导频泄漏、投影模型、候选编码与可观测性审计。

对照与结论边界：保留合成闭环和负结果；早期字段候选未经本地完整 CRC 证实。

依赖：依赖 references/qin、E001 和常量快照。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E003/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [audit_header_observability.py](scripts/audit_header_observability.py)
- [audit_nonpilot_leakage.py](scripts/audit_nonpilot_leakage.py)
- [check_paired_unknown_overlap.py](scripts/check_paired_unknown_overlap.py)
- [decode_paired_tcode.py](scripts/decode_paired_tcode.py)
- [extract_signal_content.py](scripts/extract_signal_content.py)
- [present_signal_content.py](scripts/present_signal_content.py)
- [probe_unknown_subcarrier.py](scripts/probe_unknown_subcarrier.py)
- [projected_nonpilot_decoder.py](scripts/projected_nonpilot_decoder.py)
- [refine_perframe_content.py](scripts/refine_perframe_content.py)
- [validate_projected_decoder.py](scripts/validate_projected_decoder.py)
- [verify_known_pilots.py](scripts/verify_known_pilots.py)
