# E056 · LEO Tracker 频偏分支的符号内验证

在原工程的两份 IQ 选段中检验频偏，复用其 FrameClock 和已公开的 Qin 边缘导频。原始 IQ 与选择文件从外部目录只读加载并和本地快照核对。

- [实验设计](../../docs/plans/2026-11-21-leo-cfo-branches.md) · [结果和局限](../../reports/E056/2026-11-21-LEO频偏分支与导频预测.md)
- [实测运行](../../runs/v054-leo-cfo-aperture/manifest.json) · [重复运行](../../runs/v054-leo-cfo-confirm/manifest.json)

运行 `py -3.14 tools/run_experiment.py audit_leo_cfo_branches --run-id <新编号>`。使用安装了 NumPy 和原工程可导入依赖的 Python 3.14 环境。可用 `STARLINK_EXTERNAL_LEO_ROOT` 指定外部目录。单位测试的合成样点采用种子 5601，与实测输出分开。
