# E083 · MCS 66 系统区参与项与擦除验证

保留 E081 在 MCS 66 中已验证 PN15 停止的全部末字，按填充起点逐块搜索 E082 条件校验中缺失的系统区参与项。对冻结的零综合校验，在留出末字、非末字和 250 MS/s 原始 IQ 解调路径上复验。再仅用部分校验矩阵提出单擦除补值。

- `python tools/run_experiment.py extract_mcs66_padding_sweep --run-id v081-mcs66-padding-sweep`
- `python tools/run_experiment.py recover_mcs66_systematic_checks --run-id v081-mcs66-systematic-checks`
- `python tools/run_experiment.py audit_mcs66_single_erasures --run-id <新编号>`

复跑时为各入口使用不同标识，分析脚本读取已经冻结的提取回执和矩阵。`mcs66_padding_words.npz` 的 `bits` 和 `mask` 分别为大端打包的码字和实收掩码。`mcs66_partial_checks.npz` 保存 1,024 条校验的零基坐标及同一矩阵的行打包结果。单擦除候选有独立回执，未写回录波或实收掩码。

[实验报告](../../reports/E083/2026-11-21-MCS66系统项与部分校验矩阵.md) · [末字提取](../../runs/v081-mcs66-padding-sweep/manifest.json) · [系统项恢复](../../runs/v081-mcs66-systematic-checks/manifest.json) · [擦除检验](../../runs/v081-mcs66-single-erasures/manifest.json)
