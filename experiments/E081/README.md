# E081 · Code 5 复增益转移到高阶调制区

从 E078／E080 已校验首字的实收坐标估计每帧复增益。固定 E028／E030 星座标号、E070 码字几何和 PN15 停止判据，比较全档案的格点可见性、留出的填充边界和同两级头部的高阶重复。对 16QAM 的整槽四角硬判，将两个幅度位保留为擦除后再重复关键比较。

运行 `python tools/run_experiment.py audit_code5_gain_transfer --run-id <新编号>`。输入来自固定版本秦档案页缓存及 E028、E030、E070、E074、E078、E080 的归档产物。结果保存每帧复增益、逐分配的符号和 PN 边界计数及近重复码字的逐区段差异。复增益和帧内数据可从固定档案页重算。

- [报告](../../reports/E081/2026-11-21-Code5复增益与高阶调制.md) · [首轮](../../runs/v079-code5-gain-transfer/manifest.json) · [首轮复跑](../../runs/v079-code5-gain-confirm/manifest.json)
- [最终运行](../../runs/v079-code5-gain-masked/manifest.json) · [最终复跑](../../runs/v079-code5-gain-masked-confirm/manifest.json)
