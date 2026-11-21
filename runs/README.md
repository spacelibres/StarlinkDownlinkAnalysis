# 独立运行记录

每个运行编号对应 `manifest.json`、`run.log` 和实际改变的 `artifacts/`。运行失败也保留记录。证据副本在被 Git 忽略的 `.cache/runs/`，避免重复提交全部历史数据。

通过 `python tools/run_experiment.py --list` 选择脚本。提交新的科学结果时，须审阅运行记录、填写 changelog 并递增版本；不能把合成对照通过写成真实录波解码成功。
