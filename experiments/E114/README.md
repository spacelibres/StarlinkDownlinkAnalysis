# E114 · 原 WAV 的逐帧相对符号

E113 的 160 ms 相干均值把资源位置 34 留作擦除。E114 检验其逐帧贡献是否可重复测到，以及同一载波的另外两个弱均值位置能否预测它。输入为原 WAV 和 E040–E042、E025、E113 的冻结记录，哈希逐项见结果。窗时刻沿用 E113，保留 13.6 秒前的 43 个有效窗。

沿用 E113 的三载波定时和频偏，从每帧取 3,200 个复样点，补偿分数样点时差并计算频谱。用其它三根载波校准逐帧公共相位。目标载波的拟合采用两组互斥频点（201、111），两种信道响应（常数、频率一次），并放开槽类 4、5、12。按八个连续整帧为组，在各窗内随机分配两折，名义增益用对折估计，分频组各自计算。槽类 4、5、12 对应资源模 60 的 50、34、42。四个分支同号且各自实部幅度至少为参考增益的 0.35 时才保留相对符号。符号相对 E025 模板和 E113 固定循环相位 8，擦除用 0 保留。

去掉各窗均值后，用两个频组的实部交叉乘积度量共同变化。固定错帧移位作为对照。估计的区间以最多四个相邻窗为重采样块，共 11 块、499 次有放回抽样。合成正对照在已对齐的实测频谱中注入逐帧交替的符号，幅度为局部参考增益的 0.5 或 1。该注入检查分帧估计器。独立的评分程序只用其中一组频点上的槽类 4、12 生成预测，另一组频点上的槽类 5 负责评分，双向互换。每窗各取 63 个非零循环移位，保持整帧及其符号结构。分组和重采样种子分别为 `20261121116` 和 `20261121117`，留出评分的移位种子为 `20261121117`。

LEO 对照已有可复算入口。E108 的原工程 `FrameClock` 与两份配对 `selected_iq.npz` 支持一台估相、另一台验证留出样点。经本机源文件核对，两个 NPZ 的 SHA-256 与已归档副本一致。其接收频窗在下行上边缘，WAV 当前模型用中心附近载波。要验证同一个位置，须先对齐资源坐标和极性，并确定该位置真实可观测。两种录波的相对资源编号可用于提出可检验假设。[LEO 原工程接入](../E099/README.md) · [LEO 双路留出](../E108/README.md) · [LEO FrameClock](<../leo-tracker-reduxredux-main/tools/dsx01_pilot_clock.py>)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_wav_slot_frame_power --run-id <新编号>
python tools/run_experiment.py verify_wav_slot_sign_transfer --run-id <新编号>
```

[报告](../../reports/E114/2026-11-21-WAV逐帧符号与共同状态.md) · [初次运行](../../runs/v112-wav-slot-frame-power/manifest.json) · [逐帧运行](../../runs/v112-wav-slot-frame-power-final/manifest.json) · [逐帧复跑](../../runs/v112-wav-slot-frame-power-confirm/manifest.json) · [留出运行](../../runs/v112-wav-slot-sign-transfer/manifest.json) · [留出复跑](../../runs/v112-wav-slot-sign-transfer-confirm/manifest.json)
