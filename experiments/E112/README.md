# E112 · 原始 WAV 在终止事件前后重新捕获循环字

使用原始双通道 WAV、E001 移动细线相位、E040 的帧率、E041 的二次定时修正及 E042 的四载波时域模型。参考为 E025 冻结 60 位字。前后各段保留各自起点，共 14 个互不重叠的 160 ms 窗。沿用 E042 的 12 谐波整组留出，两个方向互换训练和留出。

每窗去除已知细线频漂，以该窗中心的时钟导数修正梳谱间距。在 ±375 Hz 内以 5 Hz 遍历残余频偏，从训练谐波按线能量选三个点，候选之间及与 E090 原频偏至少间隔 15 Hz。四个候选分别搜索 60 个循环相位和一个帧周期内的 3200 个整数时延。用 FFT 计算时移后的各列内积，再求每载波复增益的最小二乘解。时延、频偏、字相位和增益全部从训练谐波确定。冻结模型预测留出谐波，用相对零预测的平方误差减少评分，两侧频带也各自评分。[Qin 等的 60 位重复模型](https://arxiv.org/html/2602.02627#S8.SS2) · [Golub 与 Pereyra](https://doi.org/10.1137/0710036)

对照在三个实际事件后时窗的去漂波形里注入合成四载波分量。种子 `20261121113`，每窗独立选模拟循环相位、频偏和整数时延，幅度参考 12.80 秒窗的实测增益。全幅、半幅及四分之一幅使用同一时窗背景。验收条件为循环相位与频偏网格点正确、时延误差至多一个样点、留出误差减少为正。分支共享背景，试验只评估模型条件下的灵敏度。原始 WAV 单独保留。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_wav_cycle_reacquisition --run-id <新编号>
```

[报告](../../reports/E112/2026-11-21-WAV终止前后循环字重捕获.md) · [首轮](../../runs/v110-wav-cycle-reacquisition/manifest.json) · [分侧正式运行](../../runs/v110-wav-cycle-reacquisition-final/manifest.json) · [复跑](../../runs/v110-wav-cycle-reacquisition-confirm/manifest.json)
