# E007 · 条件子载波提取与解码进展

条件模型能否提取中心载波；哪些未知区域仍不能解释？

输入：原 WAV、候选前缀与频率网格、跨组估计。

方法：六载波复数提取、逐槽半子载波相位修正、结构与随机对照。

对照与结论边界：保留 240 MHz 网格经滤波降采样的合成对照；没有本地未知比特/有效 CRC 头部。

依赖：依赖 E001/E005/E006；frame_candidates 只是候选片段。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E007/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [extract_central_ofdm_candidates.py](scripts/extract_central_ofdm_candidates.py)
- [present_decode_progress.py](scripts/present_decode_progress.py)
- [publish_decode_progress.py](scripts/publish_decode_progress.py)
- [test_central_binary_regions.py](scripts/test_central_binary_regions.py)
- [validate_central_narrowband_model.py](scripts/validate_central_narrowband_model.py)
