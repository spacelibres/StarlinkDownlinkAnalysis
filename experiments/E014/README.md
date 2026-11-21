# E014 · 独立UT宽带捕获

未修改作者v2发布50ms/250MS/s CI16与旧24ms/480MS/s捕获分开。原始哈希/格式和来源在 `provenance/ut_capture_import.json`。冻结E012/E013的P/H，不重新拟合编码。

前端以重复结构/已知PSS获取、SSS信道和半导频CFO/clock校正；另一半导频保留。H1字段预测H2位置，H2块数由前缀读出，包含新的五块形式，不搜索CRC。记录全部候选及失败、完整帧与资源可用部分片段。

入口 `python tools/run_experiment.py validate_ut_capture --run-id <新编号>`。下载脚本只作一次性原始输入获取，拒绝覆盖。计划见 `docs/plans/2026-11-20-independent-ut-capture.md`。不声称应用明文或原用户窄带消息已解。
