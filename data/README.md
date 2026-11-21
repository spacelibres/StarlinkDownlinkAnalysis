# 数据分类

- `raw/wav/`：用户原始复数 IQ WAV，只读；完整哈希在基线清单中。
- `raw/leo_tracker_snapshot/`：外部仓库两份选段及分析常量的只读快照；不包含外部全库。来源、大小与哈希见 `../provenance/leo_input_snapshot.json`。
- `raw/legacy_wideband/`：用户旧工程保留的公开 v1.0.0 宽带数据的 complex64 NPY 转换；逐样点核对原 CSV，来源/哈希见 `../provenance/legacy_research_import.json`。它不是用户原始窄带 WAV 的新采集。
- `raw/ut_250msps/`：作者 v2.0.0 发布的 50 ms 实测数据。文件为小端有符号 int16 交错 I/Q，与发布摘要一致，归属与哈希见 `../provenance/ut_capture_import.json`。
- `derived/ci16/`：历史 IQ 导出及 manifest；交错有符号 int16 I/Q 不是语音，也不是协议消息。
- `derived/frame_candidates/`：原始录波候选时间段的 NPZ 切片；候选位置不等于帧已确认。

用户补放的 `Starlink_Downlink_IQ_250MSps.bin` 与 `raw/ut_250msps/` 的归档字节相同。实验读归档路径，按同一份 50 ms 捕获计数。补放文件保留在原位置，核对记录见 [250 MS/s 补放副本](../provenance/ut_250msps_user_drop.json)。

250 MS/s 录波的实验入口见 [E014 双头部与 Code 5](../experiments/E014/README.md)、[E025 二元尾段](../experiments/E025/README.md)、[E043 双时轴与弱帧](../experiments/E043/README.md)和 [E044 偏移时轴导频与共同首头部](../experiments/E044/README.md)。本地补放路径由 Git 忽略，单份归档由 Git LFS 管理。

不凭文件名推断未知采集地点/日期/时区。数据变换必须记录原文件、采样率、I/Q 顺序、尺度、频率约定、起止样点和输出哈希。新增结果通过独立运行记录产生，封存文件保持不变。
