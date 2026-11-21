# E091 · Code 5 内部目录 CRC24 和首条记录跨度

对 E078 和 E080 冻结的首字先独立复算外层 Code 5 校验，再从两个档案校准长度识别内部 CRC。以其余长度、整组留出和两份原始宽带检验逐字节预测。另从 Qin 缓存恢复第一分配的第二个 Code 5 字，以固定的长度读法预测记录边界。

运行 `python tools/run_experiment.py audit_code5_directory_checks --run-id <新编号>`。归档运行将 `OPENBLAS_NUM_THREADS` 和 `OMP_NUM_THREADS` 设为 1。本地装有已登记的 Qin 页缓存时离线运行。

[报告](../../reports/E091/2026-11-21-Code5目录CRC24和记录跨度.md) · [首轮](../../runs/v089-code5-directory/manifest.json) · [补足截断字](../../runs/v089-code5-directory-final/manifest.json) · [复跑](../../runs/v089-code5-directory-confirm/manifest.json)
