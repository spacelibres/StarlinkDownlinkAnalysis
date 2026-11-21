# E135 · LEO 早期符号关系的跨录波检验

以 `../leo-tracker-reduxredux-main` 为只读来源。接续 `reports/2026_09_28_sequence_semantics/HEADER_BLOCK_RANK.md`，检验原工程保留的两条九位异或和 70 条跨符号异或能否预测新的、已归位的时频观测。这 70 条来源于原工程 37 个成对帧秩筛查窗口。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_affine_transfer --run-id <新编号>
```

默认读取上述路径，可用 `STARLINK_EXTERNAL_LEO_ROOT` 指向同一份快照。[完整报告与既有研究对应表](../../reports/E135/2026-11-21-LEO早期符号关系复核.md) · [正式运行](../../runs/v133-leo-affine-transfer-final/manifest.json) · [确定性复跑](../../runs/v133-leo-affine-transfer-confirm/manifest.json)
