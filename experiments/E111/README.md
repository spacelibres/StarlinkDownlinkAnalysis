# E111 · 弱帧体区软填充起点及留出校验

读取 E110 两份冻结的 H2 结果，只用两次一致且通过 CRC 的三个弱帧，切出九个分配的末字。重新从 250 MS/s 原始 CI16 生成实收 OFDM 观测。沿用 E044 两套频偏／时钟修正、E029 PN15 及 E028／E030 星座标号。两个拟合共享样本。

每个末字使用已验证的最后 128 个完整系统资源作锚。按 32 资源整组从四组选两组拟合复增益和噪声，另两组评分。对锚前数据再随机分配互补整组，各自拟合 PN 起点，再用对应留出组评分。种子 `20261121112` 派生到时隙和格式，两套拟合共用分组。估计采用复高斯噪声代理和均匀星座混合基线。在每个候选位界之后，把每个资源的星座似然在符合已知 PN 位的所有标签上边缘化。边界落在符号内部时仅固定属于系统区的 PN 位。移位 ±1 的 PN 作为留出对照。[Caire 等的星座似然边缘化](https://doi.org/10.1109/18.669123)

从 E085 加载冻结的 2,560 条 MCS 66 校验。在每条校验中屏蔽坐标最大的整个校验 QAM 符号，同符号有多个被检查位时先在星座标签上求其联合异或期望。可见输入的各符号期望相乘，预测被屏蔽符号的相应异或。比较仅实收软判和 PN 条件化两种方式，对近锚区之外的填充依据留出支持和额外一个符号组留余量。保留系统项移位、63 种校验行族内循环错位、三个弱 MCS 70 末字和 16 个已归档强 MCS 66 字。目标校验的观测只在评分时使用。[Kschischang 等的因子图及和积](https://www.isiweb.ee.ethz.ch/papers/arch/aloe-2001-1.pdf) · [E085 部分矩阵](../E085/README.md)

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python tools/run_experiment.py audit_ut_weak_padding_and_parity --run-id <新编号>
```

[报告](../../reports/E111/2026-11-21-弱帧填充起点及MCS66校验.md) · [运行](../../runs/v109-ut-weak-padding-and-parity/manifest.json) · [复跑](../../runs/v109-ut-weak-padding-and-parity-confirm/manifest.json)
