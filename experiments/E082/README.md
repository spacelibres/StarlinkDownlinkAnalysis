# E082 · 增益校正后的高阶校验

沿用 E081 的帧增益、幅度擦除和 E077 的整组划分。保存近全 PN 的高阶末字和 MCS 66/70 的非末字对照，复跑密集准循环折叠，再对 MCS 66/70 候选校验区搜索两项或三项的仿射异或。

使用固定版本 Qin 范围缓存。先运行 `python tools/run_experiment.py extract_gain_padded_words --run-id v080-gain-padded-words` 建立供两个分析读取的固定产物。已有固定产物时，用不同运行标识重复提取即可核查哈希。`gain_padded_words.npz` 中 `_bits` 和 `_seen` 使用大端打包，分别表示原格点判决与实收掩码。`_conditioned` 和 `_conditional_mask` 仅包含模型所需的 1,024 位前缀，保守起点之后为已知 PN15 假设。

- `python tools/run_experiment.py audit_gain_calibrated_folds --run-id <新编号>`
- `python tools/run_experiment.py audit_sparse_padding_checks --run-id <新编号>`
- `python tools/run_experiment.py audit_sparse_qc_orbits --run-id <新编号>` 读取 `v080-sparse-padding-final` 的校准候选。
- `python tools/run_experiment.py audit_raw_sparse_orbits --run-id <新编号>` 读取 `v080-sparse-qc-orbits` 的冻结轨道。

[实验报告](../../reports/E082/2026-11-21-MCS66条件稀疏校验.md) · [提取](../../runs/v080-gain-padded-words/manifest.json) · [折叠](../../runs/v080-gain-folds/manifest.json) · [稀疏检验](../../runs/v080-sparse-padding-final/manifest.json) · [旋转](../../runs/v080-sparse-qc-orbits/manifest.json) · [原始 IQ 复验](../../runs/v080-raw-sparse-orbits/manifest.json)
