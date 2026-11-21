# 复现与证据保全

## 输入和版本

`provenance/baseline_manifest.json` 列出整理前 183 个非缓存文件的路径、大小和 SHA-256；`layout_manifest.json` 逐一对应当前路径。`preserve_bytes=true` 的证据必须字节一致。改动过路径适配的脚本、Notebook 可从 `v0.0.1` 取得原版。

原 WAV 的 SHA-256 为 `af03914ea773bf2eaa2698dd674afa6ac6a5ecc9cb6a69f81350777ab5789041`。采集地点、文件名日期解释和时区未知；不补造。原始数据、CI16/NPZ/MAT/图像通过 Git LFS 管理。仅有指针时先运行 `git lfs pull`。

外部仓库的小型只读输入快照包括两份 selected IQ、选择元数据和参考常量源文件。哈希见 `provenance/leo_input_snapshot.json`。它不包括完整外部 NPZ 库；E002 全库清点/轨道审计仍读取 `STARLINK_EXTERNAL_LEO_ROOT`，默认 `../leo-tracker-reduxredux-main`。该外部来源缺少可核实的根 LICENSE，已明确记录，未擅自重新许可。

[E071](../experiments/E071/README.md) 从同一外部根目录只读核查配对录波清单及高采样率探针的调谐参数。结果保留候选录波的 `bulk://recordings/` 地址和可查的清单指纹。档案工程的 Linux 原始 IQ 存储在源码快照之外。

## 隔离运行

```powershell
python tools/run_experiment.py --list
python tools/run_experiment.py validate_cyclic_estimators --run-id my-control-001
```

入口为每次运行创建新的 `runs/<id>/`。既有结果、图表、报告、参考和派生数据复制到 `.cache/runs/<id>/workspace/`，通过兼容路径层读写副本。禁止复用运行编号。改变的产物复制到 `runs/<id>/artifacts/`，其余副本留在可再生缓存中。

清单记录版本、Git HEAD/脏状态、代码哈希、原始输入哈希、可用证据快照哈希、命令、解释器、UTC 时间、返回码与输出哈希。可用输入清单不等于实际文件访问跟踪；种子和处理参数仍以各脚本/结果记录为准。返回码非零也保留日志与产物。

这是防止误覆盖的路径管理方式，不是安全沙箱。不要绕开入口手工设置运行环境。历史脚本存在顶层执行与跨实验引用；本轮只验证指定的快速合成对照，未宣称所有历史脚本均已完整重跑。个别结果来自过去交互执行，没有独立生成脚本；原始证据和来源缺口均保留。

## 核验

`python tools/verify_project.py` 检查全部迁移项、封存证据哈希、本地依赖快照、活动 Python 语法、旧文件引用与 Notebook 路径。`python -m unittest discover -s tests -v` 检查布局及提交规则。数值对照应通过运行入口单独记录；不能用“语法通过”代替科学验证。

依赖版本见 `requirements.txt` 和 `provenance/environment_baseline.json`。不将第三方参考源代码全部当成当前 Python 版本的可执行模块。

Windows PowerShell 的 `python` 命令可能由配置文件指向与子进程 PATH 不同的解释器。可先用 `python -c "import sys; print(sys.executable)"` 确认路径，批量检查应沿用该绝对路径。实验入口内部通过 `sys.executable` 启动，避免切换到缺少 NumPy 的其他环境。

## E009 旧研究恢复

`provenance/legacy_research_import.json` 是两个用户指定旧工程的定向只读来源清单，含 47 个快照。宽带输入单独保存在 `data/raw/legacy_wideband/`，不与原 WAV 或 LEO 选段混用。原 CSV 已按历史 SHA-256 核验，NPY 的 11,520,180 个 complex64 样点与 CSV 转换完全一致。

正常复现无需原旧目录：`python tools/run_experiment.py replay_legacy_wideband` 使用本地输入/代码及原始存档对照；运行前核验导入快照。`collect_legacy_sources.py` 是一次性来源恢复入口，会拒绝覆盖已有导入，不要为普通复跑重新执行。pypdf 仅用于旧文献元数据索引，不参与数值译码。

历史数据与当前作者仓库新数据版本不同。旧数据上游目前说明暂不可下载，因此不声称它仍可从原 URL 获取；本轮未重新发布数据。当前 17 头部成功是继承既有方法的同捕获协议后端复现，非新独立采集、非用户窄带消息，后续泛化仍需另一份实测捕获。

## E010 双头部复现

`python tools/run_experiment.py replay_full_header_layout` 使用同一本地宽带输入，按完整 FFT 范围增加旧帧索引 −1，合计 18 组头部。`provenance/legacy_second_header_import.json` 另记录 9 项只读恢复的旧源码/存档；一次性导入脚本不用于日常复现。旧存档只在当前译码与控制计算完成后才读取用于比较，但历史参数仍来自相同捕获，不能视为独立验证。

封存运行 `v008-complete-header-layout-02` 保存全部 36 头部软判决、保留比特、重编码、描述符 CSV 和逐个负对照。其执行时项目版本为 0.0.7 且工作区有改动；清单给出实际代码哈希，随后经测试形成 0.0.8 里程碑。首个运行也保留；第二次仅加强了验收门，不增加独立样本数。

## E011 端点与可辨识性

`python tools/run_experiment.py audit_allocation_observability --run-id <新编号>` 先从本地宽带 NPY 算完 18 帧相干量/阈值/最小二乘及有限排列控制，再读取已封存 E010 头部作比较。主记录 `runs/v009-allocation-observability-02/` 的 JSON 同时保留无结果、原始候选和局部对比准入，NPZ 保存三种 PN 条件下的完整相干量，CSV 保存逐帧端点。

首个运行的端点/全部信号诊断未因跟进修改；第二次增加正单位可行性线性规划及对照的同一局部对比判据。它们不是独立重复实验。`ruptures` 仅为已阅读的作者参考实现，没有安装或从网络动态执行；当前 SSE/线性规划使用既有 NumPy/SciPy 依赖。公式补充与来源边界在 `learning/allocation_formulas.json`。

## E012 Code5 编码关系

入口：`python tools/run_experiment.py audit_code5_parity --run-id <新编号>`，使用本地 NPY、E010 封存头部与已有 SSS MAT。频率/采样校正参数继承论文，未由目标载荷校验优化。预定14字训练，QC128扩展后满秩才输出矩阵；秩亏或有矛盾会保存提取结果并返回2。主记录 `runs/v010-code5-parity-02/` 保存逐字表、全部观测、硬比特、训练标记、P和预测校验。

首轮单位幅度重调制误差较大，跟进只添加已知头部前48编码符号的复数增益诊断，原符号/硬比特/P/校验预测逐点不变。原尺度NMSE仍保留，不把后续校准伪装成最初预设结果。两次均为同捕获复跑；后续需要稀疏完整H和另一份实测捕获。

`provenance/legacy_payload_import.json` 保存4项旧源码/报告的只读快照，其中错误标为载荷的H2区间不予继承。一次性恢复脚本拒绝覆盖，不作为常规复跑入口。全局线性代数参考 `galois` 只阅读作者实现，未安装/执行或代替当前数值验证。

## E013 稀疏校验与软纠错

入口：`python tools/run_experiment.py audit_sparse_code5 --run-id <新编号>`。仅需E012主运行已封存NPZ，无需重新读取旧目录或重新校正RF。程序先从冻结P选取H，验证满秩/正交/逆重构后才读取接收码字；同一已知前缀尺度直接使用，不拟合译码目标。

`runs/v011-sparse-code5/` 保存搜索计数、8种子、H和逐例译码结果；源码随里程碑版本进入Git，运行清单给出其哈希。47项是项目单元测试总数，不是RF实验次数。合成检查与真实运行分开；H的所有1024稀疏行均来自精确全维搜索，未用稠密行补秩。译码器未安装或调用外部LDPC包；作者C++实现只作已读方法参考。

## E014 跨录波验证

`python tools/run_experiment.py validate_ut_capture --run-id <新编号>` 读取本地 250 MS/s 原始 IQ、E012 的 P、E013 的 H 及 Qin 的 SSS。入口核验 IQ 和矩阵摘要。重采样缓存可由原始输入重建。来源文件已随本地 Git LFS 导入，常规复跑无需下载。

主运行 `runs/v012-ut-transfer/` 与复跑 `runs/v012-ut-transfer-repeat/` 的 JSON 和 NPZ 摘要相同。通过清单定位文件可核验退出状态、产物哈希和原证据改动数。用 NPZ 的 `hard_code5_words` 分别验证 `(words[:, :1024] @ P) % 2` 与后半字逐位相等，以及 `(words @ H.T) % 2` 为零。两次运行使用同一录波。
