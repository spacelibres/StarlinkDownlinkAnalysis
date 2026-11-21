# E006 · 突发沿、同步和中心频率网格

终止沿如何对齐；窄带同步匹配依赖何种中心网格？

输入：原 WAV、轨迹、参考 PSS/SSS 与循环结果。

方法：分带包络、终止沿、相干/非相干同步探测和中心缺口网格检验。

对照与结论边界：沿精度受滤波与 SNR 限制；已知模板匹配不等于未知载荷解码。

依赖：依赖 E001/E005 和 references/qin。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E006/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [acquire_narrowband_sync_prefix.py](scripts/acquire_narrowband_sync_prefix.py)
- [measure_burst_edge_alignment.py](scripts/measure_burst_edge_alignment.py)
- [test_central_gutter_grid.py](scripts/test_central_gutter_grid.py)
- [wav_coherent_frame_probe.py](scripts/wav_coherent_frame_probe.py)
- [wav_noncoherent_sync_probe.py](scripts/wav_noncoherent_sync_probe.py)
