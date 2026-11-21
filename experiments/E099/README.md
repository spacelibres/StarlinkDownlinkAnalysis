# E099 · LEO 原工程输入索引与留出导频复算

主项目已重放 LEO 的 2.5 MS/s 短窗，也已比对 DS7 位导出。后续 Code 5 字段主要从宽带发现及验证。此次重新核查本地 LEO 目录，并检验另一路可直接复算的 DS7／DS8／DS9 上边缘导频档案。[解码证据总表](../../docs/DECODE_MAP.md)

| 原工程输入 | 本机可直接执行的研究 | 证据边界 |
|---|---|---|
| `reports/2026_10_18_dsx01_pilot_clock/` 下两份 `selected_iq.npz` | 对 10 个去重短窗重放帧同步、频漂、分数样点时轴和相位检验 | 2.5 MS/s、下边缘，已归档 [E055](../E055/README.md)、[E056](../E056/README.md)、[E089](../E089/README.md) |
| `reports/2026_09_28_pilot_split_transfer/{DS7,DS8,DS9}/matrices.npz` | 复算 7.5 MS/s 上边缘的已解调导频残余频偏和相干性对照 | 150×8 导频矩阵保留已匹配导频，原始时域样点及未知数据子载波不在此文件内 |
| `reports/2026_09_28_ds89_baseline_transfer/exports/*/banks/banks.npz` | 复用 TLE 候选、位置、速度和时间网格做轨道对照 | DS8-001、DS9-001 文件头的 193 个数组均由时间网格及候选 ID、位置、速度组构成 |
| DS7 逐位导出及 DS7–DS9 分析回执 | 用冻结宽带候选复核可读位导出，复算可得的派生矩阵 | DS7 的 38 个可读导出已对照。DS8／DS9 的指定证据缺项见 [E093](../E093/README.md) |
| 25 MS/s 配对档案 | 用现有调谐、流、计数间断信息确定真实空口重放对象 | `cap-20260902T224009-0181f7f0ffa1` 的 `stream-0` 原 IQ 是直接验证时钟的优先对象。已知原始归档依据和另外五条上边缘候选见 [E071](../E071/README.md) |

## 方法

只读原工程，保留其会话、访次、接收机、频偏拟合和每帧乱相种子。三个输入均为 7.5 MS/s 上边缘。矩阵第 0、2、…、148 行对应原工程用来拟合频偏的偶数行。重放只取第 1、3、…、149 行，以归档的零偏、普通和稳健频偏各算一次真实及乱相相干性。乱相给每个符号施加一项 QPSK 相位，同符号的八根导频共用相位。判据为每一分数的绝对差小于 `1e-12`。没有新增随机拆分，汇总保留原接收机、访次和会话；同窗帧共用采集条件。

运行前核对三个档案的矩阵、结果和规格 SHA-256，以及规格引用的基线输入。脚本计算归一化复数投影，并分别保留每个窗的留出收益和每个数据集的最大复算差。结果依赖原先同步、载波去旋和固定频偏，适用范围为频偏模型与导频观测。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_pilot_archive --run-id <新编号>
```

[结果](../../reports/E099/2026-11-21-LEO原工程接入与导频复算.md) · [全库文件头复扫](../../runs/leo-source-review-20261121/manifest.json) · [运行](../../runs/v097-leo-pilot-archive/manifest.json) · [复跑](../../runs/v097-leo-pilot-archive-confirm/manifest.json) · [原工程导频报告](https://github.com/misko/leo-tracker-reduxredux/tree/main/reports/2026_09_28_pilot_split_transfer)
