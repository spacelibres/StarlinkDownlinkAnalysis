# E145 · Code 28 缩短核心的跨录波检验

从 Qin 已校验的 H2 读取 Code 28 分配，只读归档中的均衡星座页。固定 E021 的旧 480 MS/s 生成器和码字布局，使用归档的 H1 相位和首个 Code 5 BPSK 系统区拟合复数尺度。比较硬判、整体反相、星座点擦除及增加 +511 擦除的结果。旧录波码字是回归正对照。用 E030 冻结的 PN15 单独验证新录波中的系统区停止位置。整批评估无 Qin 参数训练或跨帧调参。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code28_archive_transfer --run-id <新编号>
python tools/run_experiment.py verify_code28_archive_witness --run-id <另一新编号>
```

读取项目缓存的 Qin HDF5 页，并与 E067 的页长度、SHA-256 核对。第二个脚本以冻结的首轮结果选择唯一见证字，分七轮屏蔽其原实收校验位。每次复跑使用新编号。[报告](../../reports/E145/2026-11-21-Code28跨录波缩短核心.md) · [全档案运行](../../runs/v143-code28-archive-transfer/manifest.json) · [全档案复跑](../../runs/v143-code28-archive-transfer-confirm/manifest.json) · [屏蔽校验块](../../runs/v143-code28-witness/manifest.json) · [屏蔽复跑](../../runs/v143-code28-witness-confirm/manifest.json)
