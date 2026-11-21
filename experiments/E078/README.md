# E078 · Code 5 的字节偏移和可选短记录

在两份原始宽带以及秦档案内，译码 H1 指向的 Code 5 起始字，用冻结 PN15 读出系统字节。检验首字节偏移、7 字节记录和 28 字节可选区，保留原始及擦除重建的逐位证据。

运行 `python tools/run_experiment.py audit_code5_byte_records --run-id <新编号> -- --variants-and-erasures`。依赖 E012/E014/E036/E073 归档及固定版本 Qin 缓存。最终 NPZ 中 `frame_<编号>_word_<序号>_bits` 和 `…_observed` 分别是大端打包的 2,048 个码字位与实收掩码。系统字节需应用冻结 PN15 后按小端位序逐字节打包。

- [计划](../../docs/plans/2026-11-21-code5-byte-records.md) · [报告](../../reports/E078/2026-11-21-Code5字节位置与短记录.md)
- [首轮](../../runs/v076-code5-byte-records/manifest.json) · [完整运行](../../runs/v076-code5-byte-words/manifest.json) · [复跑](../../runs/v076-code5-byte-words-confirm/manifest.json)
