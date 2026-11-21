# 公式与可执行学习

`formula_library.json`、原公式 Notebook 与合成编码教学保留最初来源和字节。原库为 50 项公式卡、20 条来源记录；不是全球穷尽目录，也不是全部已实测验证。

后续专题补充独立存放，避免改写旧证据：

- `allocation_formulas.json`：E011 新增 5 张资源相干量、L2 变点、逐码字取整、单位可辨识性和数量/尺度简并卡。各卡区分原论文、作者实现、当前复现及局限。
- `parity_formulas.json`：E012 新增 6 张接收校正、GF(2)生成关系、QC结构、秩限制、全字取反和前缀尺度卡，逐项联系真实数据检验与未解决的问题。
- `sparse_decoder_formulas.json`：E013 新增 5 张稀疏对偶、满秩等价性、奇偶逆重构、LLR及归一化min-sum卡；明确原码字零综合与真实纠错证据的区别。
- `educational_fec_demo.py`：合成真值下的编码/译码教学。不能拿它代替真实未知载荷证据。

完整参考登记在 `references/research_register.json`；真实实测结论见 `docs/CURRENT_STATUS.md`。来源条目数、公式数和代码仓库数都不等于独立实验次数。
