# E001 · 基础录波、频谱与事件

录波中哪些频谱分量随时间演化，终止事件是否跨分带一致？

输入：原始 WAV；频谱、轨迹与事件数组。

方法：STFT、谱峰/轨迹、事件分频带功率。

对照与结论边界：窗口与频带影响分辨率；谱形本身不足以确认协议字段。

依赖：E004–E007 使用本组轨迹或频谱。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E001/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [analyze_iq.py](scripts/analyze_iq.py)
- [deeper_checks.py](scripts/deeper_checks.py)
- [focus_event.py](scripts/focus_event.py)
- [inspect_waterfall.py](scripts/inspect_waterfall.py)
