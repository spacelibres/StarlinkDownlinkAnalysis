# E107 · LEO 原生 25 MS/s 的实测频偏与时钟锚擦除

读取原工程 `0181` 的 89 个原生 25 MS/s GLRT 保留窗。相对于原工程给定的导频中心计算捕获频偏与跟踪频偏，分别加到 E071 的标称载波中心。以绝对基带频率小于 12.5 MHz 判断中心覆盖，保持实际滤波响应未知。用分数样点时刻与 750 Hz 网格求相对接收序号。每个支持窗保持原编号和样点地址。

对 E097 的 `L=32` 时钟结构，列举被擦除的锚位，不改其余接收位。只保留 15 秒区间对齐、半秒标志一致及计数在有效范围的候选。一解才输出时间，多解及无解保留状态。两个最高锚位对应的 GPS 整秒权重是 $2^{32}$ 和 $2^{33}$，它们模 15 余 1 和 2，四种组合的余数互异。四位擦除另设压力分支，允许真实的同余多解。

另从历史 480 MS/s 实测 IQ 生成超出原 E098 偏移扫描的五种 25 MS/s 波形。目标偏移由 `0181` 的捕获频偏最小值、中位数和最大值确定，用 E098 标称分支的粗频偏中位数补偿宽带原录波自身偏移。独立从限带样点同步、判决，比对同一宽带的 GPS 标签。保留错误极性、未擦除硬判和四位擦除对照。无随机拆分。全部分支及复跑共享同一次宽带录波。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_leo_native_clock_coverage --run-id <新编号>
```

[报告](../../reports/E107/2026-11-21-LEO实测频偏与时钟擦除.md) · [初次运行](../../runs/v105-leo-native-clock-coverage/manifest.json) · [正式运行](../../runs/v105-leo-native-clock-coverage-final/manifest.json) · [复跑](../../runs/v105-leo-native-clock-coverage-confirm/manifest.json)
