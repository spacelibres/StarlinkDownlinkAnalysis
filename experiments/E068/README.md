# E068 · H2 描述符布局与资源长度可辨识性

固定 E036 的 CRC 头部位串，再以 E067 的头部盲测终端相位和粗测起点检验候选拆分。用模约束给出不可能解的证书，仅用拟合档案段学习单位后预测留出段。

- [实验计划](../../docs/plans/2026-11-21-descriptor-layout-identifiability.md) · [实验报告](../../reports/E068/2026-11-21-H2描述符布局与长度.md)
- [首轮序列化失败](../../runs/v066-descriptor-layout/manifest.json) · [最终运行](../../runs/v066-descriptor-layout-final/manifest.json) · [复跑](../../runs/v066-descriptor-layout-final-confirm/manifest.json)
- [逐模型证书与预测](../../runs/v066-descriptor-layout-final/artifacts/experiments/E068/results/descriptor_layout_identifiability.json)

`python tools/run_experiment.py audit_descriptor_layout --run-id <新编号>`，使用归档的 E036 和 E067 结果，无需额外下载波形。
