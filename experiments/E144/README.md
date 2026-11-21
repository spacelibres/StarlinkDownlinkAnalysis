# E144 · LEO DS9 的 H1 局部卷积译码

使用 LEO 原工程两份双路早期符号判决 JSON。先按已确认的资源图定位 H1，再在 24 个可见编码坐标的投影上枚举 10 个输入位和全局 BPSK 极性。公共前缀仅用于事后核对。保留原工程的变量掩码分支、两路单独分支、双路同号分支以及对每个访次的 23 种同维循环错位。没有以观测结果重新拟合组别、码参数或频点。七访次的 Code 5 投影秩作为另一层接收约束。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_projected_h1 --run-id <新编号>
```

默认从 `../leo-tracker-reduxredux-main` 只读读取，也可用 `STARLINK_EXTERNAL_LEO_ROOT` 指向同一快照。[方法和结果](../../reports/E144/2026-11-21-DS9原生窄带H1局部译码.md) · [运行](../../runs/v142-leo-projected-h1/manifest.json) · [复跑](../../runs/v142-leo-projected-h1-confirm/manifest.json)
