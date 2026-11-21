# E036 · Qin 全档案头部、资源和新 PN15 边界

117 帧核验扩展到作者档案的全部 1,009 帧。独立定位目标帧尾段后，检查 MCS 6、35、42、93 并检验有限的旧编码转移。

离线重跑校验用 `python tools/run_experiment.py audit_qin_low_codes --run-id <新编号>`。源档案提取用 `survey_qin_headers` 和 `extract_qin_low_mcs`，依赖 h5py 3.16.0 和 E034 的分段来源约定。[实验报告](../../reports/E036/2026-11-21-Qin全档案与四种PN15边界.md)
