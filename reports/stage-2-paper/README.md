# Starlink Ku 下行第二阶段研究 · Stage II

**stone** · [vistone868@gmail.com](mailto:vistone868@gmail.com) · 2026-11-23

[中文 PDF](starlink-ku-downlink-stage-ii-stone-zh.pdf) · [中文稿源文件](manuscript.zh.md) · [English PDF](starlink-ku-downlink-stage-ii-stone.pdf) · [English source](manuscript.md) · [修订稿发布页 / Revised release](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/tag/paper-revision-2026-11-23)

两份全文各包含九幅图、八张表、九个公式和两段算法。图内共用英文技术标签，中文正文逐图给出完整图注。 [中期论文中文 PDF](../interim-paper/starlink-ku-downlink-interim-stone-zh.pdf) · [中期中文源稿](../interim-paper/manuscript.zh.md) · [中期英文初版](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/tag/paper-2026-11-22)

Both editions contain nine figures, eight tables, nine equations and two algorithms. They share the same measured values and figure panels.

## 结果 · Results

| 范围 · Scope | 实测结果与限制 · Measured result and limit |
| --- | --- |
| 四种已知整码字 · Four whole-word formats | 41,169 / 43,040 accepted and 2,361 / 2,635 allocations closed |
| 四种局部核心 · Four partial-core formats | 2,184 / 2,792 accepted and 763 / 952 systematic allocations closed. Low-address parity remains unresolved |
| MCS 57 | Eight additional directory CRC-24 passes. All 286 repeated-observation targets retain erasures |
| 512 位投影 · 512-bit projections | 28 / 28 verified high-address controls pass. None of the 98 low-address candidates validate |
| 250 MS/s 弱帧 · Weak events | 37 / 51 H1 and 26 / 51 H2 in the union event cohort. Two weak H2 allocation candidates lack CRC-8 confirmation |
| VHF 独立检验 · Separate VHF check | +1.08 Hz differential Doppler residual for one packet pair |

## 公开复核 · Public checks

From the repository root with Python 3.11 or later:

    python reports/stage-2-paper/verify_vectors.py
    python -m pip install -r reports/stage-2-paper/requirements-paper.txt
    python reports/stage-2-paper/build_figures.py
    python reports/stage-2-paper/render.py --language en
    python reports/stage-2-paper/render.py --language zh
    python reports/stage-2-paper/render.py --paper interim --language zh
    python reports/stage-2-paper/verify_package.py

第一步仅用 Python 标准库。它检查 6 个实测头部 CRC-8、3 个 H2 起点、8 个新增目录 CRC-24，以及从四种格式选出的 8 个核心各 4,096 条校验方程。故意改动的向量必须失败。绘图程序读取公开的汇总 JSON，无需 IQ。中文版在 Windows 使用微软雅黑，可用 `--cjk-font` 和 `--cjk-bold` 指定支持的 TrueType 字体。其他系统的字体可影响分页。

The first command needs only the standard library. It evaluates six measured CRC-8 headers, three H2 offsets, all eight new CRC-24 directories and 4,096 check equations for each of eight exemplar cores. Altered controls must fail. The figure script consumes the compact public evidence JSON. The two PDF commands use the packaged figure panels and approved identity assets. Fonts may change pagination across platforms.

## 数据与证据 · Evidence and data

| 文件 · File | 内容 · Contents |
| --- | --- |
| [evidence_summary.json](data/evidence_summary.json) | Nine panel datasets, denominator notes and 37 source SHA-256 pins |
| [verification_vectors.json](data/verification_vectors.json) | Measured header bytes, directory bytes and selected partial-core check inputs |
| [figures](figures) | Nine pairs of individual vector charts in SVG and PDF |

The manuscript cites the underlying experiment identifiers. The summary JSON provides their exact archive paths and SHA-256 values. The raw 250 MS/s, 480 MS/s and Qin archives and the complete later local run directories are outside this compact publication. Rebuilding figures and checking the extracted vectors reproduce the published presentation and selected algebraic checks. Running the receiver from IQ additionally requires the original input files and full experimental archive.

论文所引的实验编号对应底层测量记录。汇总 JSON 保留来源路径和哈希。公开版提供图表复算与选定协议向量的代数复核。完整的 IQ 接收机回放还需要原始采样和完整实验档案。

| 实验 · Experiments | 对应证据 · Evidence |
| --- | --- |
| E010, E014, E036, E091, E174 | Wideband headers, observed control exports and directory CRC convention |
| E206, E237 | VHF preamble and differential Doppler comparison |
| E218, E220, E236 | Weak H2 boundaries, ordinary wideband headers and narrowband projections |
| E229, E240–E244 | Whole-word recovery baseline and channel-estimation passes |
| E245, E246 | MCS 57 receiver, CRC directories and same-group repeats |
| E247, E248 | Partial high-address cores and masked 512-bit projections |

Public source captures and reference articles are linked in both manuscripts. Contributor results are welcome through GitHub Issues or email. Include input hashes, masking conventions, the whole-group calibration and holdout assignments, comparable wrong-coordinate controls, and retained erasures. For weak H2, include the detected offsets and CRC or waveform agreement.

欢迎在 GitHub Issues 或通过邮箱提交复现。可优先验证 MCS 57 的编码约束、部分格式的低地址校验、两帧弱 H2，或对独立 VHF 样本重做差分估计。请随结果附上输入哈希、掩码、整组留出分配、反证对照和未恢复的位置。
