# E084 · MCS 66 扩展校验矩阵

将近全 PN 末字的前 1,024 位作为可变输入，与候选校验区一起寻找低重量投影。扣除 E083 已知行的张成空间，用填充起点和实收非末字区分系统位候选。最后用留出末字、非末字及已有原始 IQ 解调复验。

- `python tools/run_experiment.py search_mcs66_early_projections --run-id v082-mcs66-early-projections`
- `python tools/run_experiment.py extend_mcs66_parity_matrix --run-id v082-mcs66-expanded-checks`
- `python tools/run_experiment.py audit_mcs66_rotation_differences --run-id <新编号>`
- `python tools/run_experiment.py audit_mcs66_expanded_erasures --run-id <新编号>`

复跑用不同标识，上游读取冻结清单。`mcs66_expanded_checks.npz` 保留原 1,024 行和追加的 1,024 行，`row_support` 为零基坐标，`check_matrix_packed` 为按行大端打包的矩阵。两版单擦除读值均保留在各自回执。

[实验报告](../../reports/E084/2026-11-21-MCS66扩展校验与PN等价解.md) · [稀疏搜索](../../runs/v082-mcs66-early-projections/manifest.json) · [扩展矩阵](../../runs/v082-mcs66-expanded-checks/manifest.json) · [旋转差分](../../runs/v082-mcs66-rotation-differences/manifest.json) · [擦除对照](../../runs/v082-mcs66-expanded-erasures/manifest.json)
