# 旧研究来源快照

按用户指定读取两个工程：

- `LOCAL_USER_DIRECTORY/Documents/starlink_beacon_iq-master/starlink_beacon_iq-master/`
- `../leo-tracker-reduxredux-main/`

原工程保持只读。`old_project/` 保存与 Ku 头部直接相关的历史脚本、报告、原始数据说明与产物；`leo_tracker/` 保存当前科学边界、证据账本和研究规则。`starlink_simulator/` 保存原作者 MATLAB 工具及原 MIT 许可。未将全部旧目录复制进新工程。

`../../provenance/legacy_research_import.json` 记录 47 个字节一致快照、17 份本地 PDF 的来源/页数/哈希/首页元数据及 10 个代码包位置。PDF 未重复复制；这些数量不是独立论文、解码器或独立实验的数量。元数据索引不表示全文结论已复现。

宽带 NPY 在 `../../data/raw/legacy_wideband/`。其 11,520,180 个 complex64 样点已经与旧 CSV 逐样点比对，差异为 0；原 CSV SHA-256 对应历史 v1.0.0 发布。新 GitHub README 已使用不同的 50 ms/250 MS/s 数据，不能混用帧名、头字或采样参数。

旧脚本包括文件创建和覆盖逻辑，因此保留为来源材料，不直接在原目录运行。E009 是明确继承旧方法的当前实现。旧报告的强结论、参数选择及撤回记录均保留，不自动提升为当前已验证事实；尤其是相同捕获的冻结复跑不等于独立空口验证。

E010 的 `second_header/` 另保存 9 项第二头部源码/存档，清单为 `../../provenance/legacy_second_header_import.json`。两次定向恢复合计 56 项材料；第二次未覆盖第一次快照。当前执行代码在 `experiments/E010/`，此处只保留来源。

E012 的 `payload/` 再保存4项旧源码/报告，清单为 `../../provenance/legacy_payload_import.json`，目前合计60项定向来源快照。`68_ku_payload_equalised.py` 把10468起的456资源误标为Code5，该段实际为H2；E012不继承此标签或其旧星座结论。新编码矩阵在当前IQ上重新求得，没有把未找到的作者矩阵当成已执行依赖。
