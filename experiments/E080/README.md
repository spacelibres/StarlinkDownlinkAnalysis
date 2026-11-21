# E080 · Code 5 档案首字恢复

把档案的逐字双点星座轴估计和窄边缘擦除转移到 E078 未恢复的首字，使用冻结码矩阵和两份校验方程复核。然后用 E079 的独立到达时轴与冻结计数关系检验新字，比较同内容条目的实收码位。

运行 `python tools/run_experiment.py audit_archive_code5_recovery --run-id <新编号> -- --complete-and-link`。使用固定版本的 Qin HDF5 页缓存和 E012、E036、E073、E074、E077–E079 的归档产物。数字档案中仅新增 479 个字，以 `frame_<编号>_word_0_bits` 和 `frame_<编号>_word_0_observed` 保存大端打包的 2,048 位补全结果与实收掩码。两个额外恢复分支及其理由单独记录。原有 530 个字复用 E078 的证据。

- [报告](../../reports/E080/2026-11-21-Code5档案首字恢复与重复内容.md) · [首轮](../../runs/v078-archive-code5/manifest.json)
- [最终运行](../../runs/v078-archive-code5-links/manifest.json) · [最终复跑](../../runs/v078-archive-code5-links-confirm/manifest.json)
