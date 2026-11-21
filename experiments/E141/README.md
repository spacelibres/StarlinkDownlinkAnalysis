# E141 · Code 5 的内部计数、尾长与窄带掩码重放

用 E101 的记录起点作为入口，从连续解扰 Code 5 字节读出内部计数、结束标记与尾长。解析器接收的是从该起点到存档连续码字末端的缓冲区。E100 给出的目标长度与下一记录位置仅用于对比。通过前缀选择旧版或新版布局，空记录独立处理。E140 先前需按已知长度保留末两个 12 位值，本次改用前置记录自己的单元数来限制清零范围。[E100](../E100/README.md) · [E140](../E140/README.md) · [宽带字段与码字来源](https://arxiv.org/abs/2610.29387v2)

沿用 E138 的完整 53 帧周期划分与种子 `20261121138`。位坐标从记录开始按低位优先读。报告同时列出全部输入字段实收的例子和部分输入经过已验证码字恢复的例子。旧版公式及对照用独立 480 MS/s 的同一份捕获。限带重放只给记录起点与 `10 e0` 格式，读取实际接收与理想频窗的交集，再用读出的计数定位结束标记和尾长。它使用 Qin 档案的实测判决。[Qin 数据来源](https://www.nature.com/articles/s44459-026-00075-6) · [E100 频窗来源](../E100/README.md)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_code5_record_extents --run-id <新编号>
```

[完整报告](../../reports/E141/2026-11-21-Code5内部长度和终止标记.md) · [正式运行](../../runs/v139-code5-record-extents-final/manifest.json) · [确定性复跑](../../runs/v139-code5-record-extents-confirm/manifest.json)
