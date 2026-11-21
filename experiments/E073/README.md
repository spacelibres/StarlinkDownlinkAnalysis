# E073 · PN15 系统位与接收校验

用两份宽带的原始硬判、冻结 PN15 及 Code 5、9、21、37 的生成矩阵检验系统区匹配段是否影响实测校验位。保留矩阵或接收标号的历史训练范围，对照修改系统位，保留接收到的校验位。

运行 `python tools/run_experiment.py audit_pn_parity_order --run-id <新编号>`。入口会校验输入归档指纹并写入独立运行结果。

- [计划](../../docs/plans/2026-11-21-pn-parity-order.md) · [报告](../../reports/E073/2026-11-21-PN15与纠错编码顺序.md)
- [运行](../../runs/v071-pn-parity-order/manifest.json) · [复跑](../../runs/v071-pn-parity-order-confirm/manifest.json)
