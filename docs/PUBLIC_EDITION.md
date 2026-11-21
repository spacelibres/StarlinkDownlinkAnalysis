# 公开版复现说明

公开版来自 StarlinkBeacon `v0.0.145`（`047674f1158591c22fb0b713e0536b2bded10ca5`）。论文注明的科学截止版本是本地档案 `v0.0.144`。历史运行的 `project_version`、Git 提交和 SHA-256 指向原档案。公开清单给出本版各文件的 SHA-256，以及经路径规范化的文本和精简运行记录在源档案的 SHA-256。

## 数据按需取得

| 输入 | 可公开的来源和完整性记录 |
|---|---|
| UT 250 MS/s，50 ms 复数 IQ | [发行方的原始下载](https://github.com/misaka666/Starlink-Downlink-Analysis/releases/download/v2.0.0/Starlink_Downlink_IQ_250MSps.bin)，放入 `data/raw/ut_250msps/`，该 50,000,000 字节文件的 [SHA-256 和格式](../provenance/ut_capture_import.json) 在库内 |
| 旧版宽带 | [v1.0.0 发行页](https://github.com/misaka666/Starlink-Downlink-Analysis/releases/tag/v1.0.0) 与 [导入记录](../provenance/legacy_research_import.json)。历史实验中的 NPY 是从 CSV 转换而来 |
| Qin 等人的补充档案 | [论文数据 DOI](https://doi.org/10.5281/zenodo.21404931) 与 [UT Austin 下载页](https://rnl-data.ae.utexas.edu/datastore/supplementaryMaterial/qin-starlink-pilots/)。参考模板及其许可见 [Qin 目录](../references/qin/) |
| StarLoc OBS0 | [Zenodo 数据集](https://zenodo.org/records/19043854) 与 [本项目选段参数](../data/raw/starloc_obs0/selection.json) |
| LEO Tracker | [外部项目](https://github.com/misko/leo-tracker-reduxredux) 与 [快照来源、预期路径和 SHA-256](../provenance/leo_input_snapshot.json)。完整外部工程可通过 `STARLINK_EXTERNAL_LEO_ROOT` 指定 |
| 用户 WAV、其 IQ 切片与 CI16 | 使用具备访问权的原始录波和所需输出。公开文档用 `data/raw/wav/USER_IQ.wav` 作为 WAV 入口。来源散列见运行清单 |

复现需按每项实验的文档补齐输入与前序产物。部分历史链接和脚本会指向原档案保留的结果及运行日志。原始 IQ、原始输出和日志、恢复字值、未核准第三方再分发材料以及完整的源 Git 历史均在首次发布范围之外。保留的运行清单是字段投影，每个对应产物的哈希仍可用于核验原档案。参考代码及数据沿用相应目录中的上游许可。

## 核验

- `python tools/verify_public.py` 检查快照文件的 SHA-256、字节数、Python 语法和运行清单数量。
- `python tools/verify_project.py` 是完整源档案核验工具。需要补齐原档案资源才可运行。
- `tests/` 包含原有测试。可独立运行 `test_release_guard.py`、`test_code28_corner_constraints.py`、`test_code5_blind_extents.py`、`test_leo_projected_h1.py`。
- `python tools/install_hooks.py` 配置变更日志、严格补丁版本和 LFS 检查。下一个公开提交由 `tools/release.py` 准备，先核验再提交并建立同版本注释标签。

## 本地磁盘占用

原始工程的 `.cache/runs/<run-id>/workspace/` 是隔离运行工作区。同一输入和旧结果被多个运行复制。按运行 ID 检查 `runs/<run-id>/manifest.json`，确认关键产物及哈希已归档、增量可复现后，再处理对应的工作区。Qin 分页缓存、预实验和独立缓存可能包含未归档的证据，需逐项核对。保留原始输入、Git LFS 对象和正式运行记录。

