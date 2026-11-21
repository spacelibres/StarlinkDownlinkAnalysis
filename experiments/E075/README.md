# E075 · 4QAM 槽量化对 16QAM 幅度位的影响

从 Qin 档案读取 E074 的近重复码字、原作者槽星座估计和限定软符号范围。记录丢失的幅度位，比较新掩码与独立保留载波上软幅度判决的一致性。

运行 `python tools/run_experiment.py audit_qin_amplitude_erasure --run-id <新编号> -- --magnitude-decisions`。输入为 E028/E074 的归档及固定版本的 Qin 范围缓存。当前所需范围与各段 SHA-256 可在最终结果的 `source.pages` 查到。省略最后的脚本参数可复现首轮距离对照。

- [实验设计](../../docs/plans/2026-11-21-qin-amplitude-erasure.md) · [报告](../../reports/E075/2026-11-21-档案幅度量化与软判.md)
- [首轮](../../runs/v073-qin-amplitude-erasure/manifest.json) · [最终运行](../../runs/v073-qin-amplitude-final/manifest.json) · [复跑](../../runs/v073-qin-amplitude-final-confirm/manifest.json)
