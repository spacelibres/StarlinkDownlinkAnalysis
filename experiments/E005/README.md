# E005 · 循环结构与周期波形

分带循环族是否可靠；能否恢复周期平均结构？

输入：原 WAV、E001 频漂轨迹与分带数组。

方法：循环估计、去频漂比较、周期波形恢复。

对照与结论边界：保留已知周期和平稳有色噪声对照；约 750 Hz 特征不是协议字段。

依赖：依赖 E001；validate_cyclic_estimators 为快速合成对照。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E005/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [analyze_subband_cyclostationarity.py](scripts/analyze_subband_cyclostationarity.py)
- [present_cyclic_evidence.py](scripts/present_cyclic_evidence.py)
- [recover_cyclic_waveform.py](scripts/recover_cyclic_waveform.py)
- [summarize_cyclic_evidence.py](scripts/summarize_cyclic_evidence.py)
- [validate_cyclic_estimators.py](scripts/validate_cyclic_estimators.py)
