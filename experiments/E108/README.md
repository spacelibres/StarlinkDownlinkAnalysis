# E108 · 用原始 LEO IQ 细化重复字和检验相位转移

使用原工程两份 `selected_iq.npz` 的两个配对窗和 `FrameClock`。沿用 E066 的帧组、原始样点掩码、模四载波映射与三组留出符号块。初始候选只使用 LEO IQ 和已知导频。前一轮的 60 位参考只在两次盲运行一致后由独立程序读取。

对于一帧和一个循环相位，先按 E066 的校准样点消去导频、前后槽泄漏和固定 PN 基线。给定正负号，按复最小二乘估计每根非导频增益，并逐帧选取两台接收机校准得分之和最大的相位。改进目标是接收机及帧各自的校准残差能量比例之和。先固定增益和相位做二元二次型的单比特下降。没有可降方向时，逐一翻转 60 个位置，每次重新剖面化增益和搜索循环相位。最多八轮外迭代。只保留校准目标严格改善的候选。

在另一整组帧中，只用校准符号块选择相位和增益。双向转移时只以对方接收机决定相位，被测接收机在固定相位用自己的校准样点估计复增益，再预测自己的留出 IQ。相位并列时保留全部并平均评分。对照为相同资源映射及 99 个保留各模四组正负计数的置乱字，使用相同拟合预算。置乱种子 `20261121108`。E066 的双帧组划分按既有清单继承。两种保护量和相互反转的折共用同一录波。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
py -3.14 tools/run_experiment.py audit_leo_profiled_word --run-id <新编号>
py -3.14 tools/run_experiment.py compare_leo_profiled_word --run-id <新编号>
```

比较脚本核验本阶段两份正式盲归档的结果一致，然后加载 E025 的冻结字。其报告按模四组取反和 60 种循环比较，不用宽带参考改写盲词或消除擦除。[报告](../../reports/E108/2026-11-21-LEO双路原始IQ重复字复核.md) · [固定相位和增益的首轮](../../runs/v106-leo-profiled-word/manifest.json) · [正式盲运行](../../runs/v106-leo-profiled-word-final/manifest.json) · [盲复跑](../../runs/v106-leo-profiled-word-confirm/manifest.json) · [宽带比较](../../runs/v106-leo-profiled-compare/manifest.json) · [比较复跑](../../runs/v106-leo-profiled-compare-confirm/manifest.json)
