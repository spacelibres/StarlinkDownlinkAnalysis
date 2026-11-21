# E121 · LEO DS9 双接收机重复字与宽带模型交叉验证

直接读取 `../leo-tracker-reduxredux-main` 的 DS9 两访次位判决回执。检验 E025 冻结宽带模板与原工程生成式的坐标是否可以对应，以及该对应是否复现每个合格窗口的完整 60 位字和相位索引。

执行 `python tools/run_experiment.py audit_leo_ds9_word_transfer --run-id <新编号>`。可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定同一快照。原工程的生成式、两份位回执和 E025 校验回执均记录 SHA-256。硬判回执中的软符号 SHA-256 作为预期值保留，软符号档案暂缺。

- [方法、结果与证据边界](../../reports/E121/2026-11-21-LEO-DS9双访次重复字.md)
- [首跑](../../runs/v119-leo-ds9-word-transfer/manifest.json)与[复跑](../../runs/v119-leo-ds9-word-transfer-confirm/manifest.json)
