# E054 StarLoc 三路同步窄带录波对照

- 本实验离线使用 StarLoc `obs_0`。源 ZIP 只取尾部 ZIP64 目录和三路未压缩 IQ 的固定字节区间。检查 HTTP 206、`Content-Range`、成员格式、尺寸和本地头部。归档每路每窗连续的原始 float32 IQ 字节及 SHA-256。保留公开原始文件时间、11.325 GHz 中心、2 MS/s 采样率和区间相对坐标。ZIP 成员的 CRC 来自目录，整成员未下载校验。无新增射频发射或采集。[数据](https://zenodo.org/records/19043854) · [读取代码](https://github.com/ConnectedSystemsLab/starloc_preproc)
- 固定选 17 个 100 ms 窗。五个窗外参考在 8、38、98、139、245 s。作者窗口内按 `50–85`、`160–195`、`195–205`、`205–230 s` 分组，后两组保留是否落在原始重叠窗口的信息。选择不依赖实际下载的 IQ。窗外参考可能仍接收到卫星信号，分别报告。[作者事件窗口](https://raw.githubusercontent.com/ConnectedSystemsLab/starloc_preproc/main/starlocProcessRawSignal/obs_periods.json)
- 核查约 43.9455 kHz 九音间隔，用单接收机搜候选，再把频率转移给另外两路。对比错间隔、同时刻左右频带和窗外样点；检验载波是否随时间同向漂移。固定主搜索及同条件的错误网格。另复用 E047 1–4 帧与半帧对照，比较全带、排除窄音的中心外侧和纯窄音的频带。三路接收是同一采集时刻的独立接收视图，不计作三个传输事件。[StarLoc 论文](https://arxiv.org/html/2604.21147v1) · [公开九音测量](https://people.engineering.osu.edu/media/document/2025-08-06/kassas_unveiling_starlink_for_pnt.pdf)
- 并排复核同一九音统计在原 WAV 和 NPZ 的结果。若某频率端的样点落出录波通带，报告不可测。事件窗口、参考校准信号及文献用于解释证据范围，不按轨道指定卫星身份。

初次导入误把尾部抓取开始字节当成 ZIP64 中央目录起点，导致最后一个 IQ 成员校验失败。使用目录自身的真实起始坐标做成员边界核对后完成导入。实验前的频谱诊断表明 20 ms Welch 平均扩散较快的窄线，采用 100 ms 相干去二次相位复核。冻结 −4,500 到 +4,500 Hz/s 的 750 Hz/s 网格和 6 个间隔，RX0 各自寻优后转移给另两路。所有窗口保留。原 WAV 取 E040 原有拟合、留出次序各前八个及全部九个事件后块的前 100 ms，独立的时间半窗作相同搜索与验证；NPZ 检查全部十个唯一 20 ms 窗。三者相干积分能力不同。
