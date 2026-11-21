# E002 · 数据清点与来源审计

两份选段和外部 NPZ 包含什么；CI16 导出是否可追溯？

输入：本地 selected IQ 快照；完整外部库未复制。

方法：NPZ 清点、复数样点解释、CI16 导出、来源和早期编号审计。

对照与结论边界：历史编号审计不是空口编号解码；全库清点仍依赖外部源。

依赖：完整清点依赖 STARLINK_EXTERNAL_LEO_ROOT。

参数、种子、频率/相位约定以原脚本与结果记录为准。缺失的历史设置不补造。部分产物由交互执行产生，没有独立生成脚本；分类不等于完整流水线已独立重跑。

`results/` 保存数值证据，`figures/` 保存已有图表；原报告（若有）在 `../../reports/E002/`。[机器可读记录](experiment.json)列出本组文件，[迁移表](../../provenance/layout_manifest.json)记录旧路径与哈希。原代码封存在 `v0.0.1`。

通过项目根目录 `python tools/run_experiment.py <脚本名>` 创建独立运行，不直接覆盖封存结果。

- [export_npz_iq.py](scripts/export_npz_iq.py)
- [identify_satellite_audit.py](scripts/identify_satellite_audit.py)
- [inventory_npz.py](scripts/inventory_npz.py)
