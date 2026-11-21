# E004 · 全局分量和频漂关联

共享频漂能否排除共轭副本或全通道增益变化？

输入：原 WAV、E001 轨迹与频谱。

方法：正反频漂补偿、谱轴、五参考轴相位对照、分带终止比较。

对照与结论边界：历史每参考轴 400 组相位对照；共享演化仍不足以区分发射共源与接收共模。

依赖：依赖 E001；坐标约定以各脚本/结果为准。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E004/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [check_global_spectral_axis.py](scripts/check_global_spectral_axis.py)
- [global_signal_audit.py](scripts/global_signal_audit.py)
- [present_global_hypothesis_tests.py](scripts/present_global_hypothesis_tests.py)
- [test_sideband_phase_coupling.py](scripts/test_sideband_phase_coupling.py)
- [test_wideband_drift_link.py](scripts/test_wideband_drift_link.py)
