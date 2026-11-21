# E034 · 公开档案头部与 Code 21 转移

对 Qin 等发表的频域符号档案，检查双头部 CRC、与原始录波的重叠及旧 Code 21 生成器的跨录波复算。

离线复算用 `python tools/run_experiment.py audit_qin_exemplars --run-id <新编号>`。已有档案切片通过 Git LFS 保存，重跑分析使用本地文件。重新从作者档案提取需 h5py 3.16.0 和网络，使用 `python tools/run_experiment.py snapshot_qin_exemplars --run-id <新编号>`。本地报大学站点证书链不受信时，已归档执行使用 `-- --allow-unverified-source`，来源和范围记录在回执中。[实验报告](../../reports/E034/2026-11-21-Qin档案与Code21转移.md)
