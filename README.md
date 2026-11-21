# Starlink Downlink Analysis

An evidence-based study of recorded Starlink Ku-band downlink signals, from waveform and frame structure to error-control coding and the recovery of selected control records.

这是 StarlinkBeacon 的精简公开版。论文和实验记录保留了数据来源、对照组、重复验证以及仍有歧义的结果。这里的控制记录恢复结论适用于报告中的录波和条件。

- [stone 署名版论文 PDF](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/download/paper-2026-11-22/starlink-ku-downlink-interim-stone.pdf) · [发布页](https://github.com/spacelibres/StarlinkDownlinkAnalysis/releases/tag/paper-2026-11-22)
- [早期英文稿](reports/interim-paper/manuscript.md) · [早期中文摘要与成果归因](reports/interim-paper/中文摘要与贡献对照.md)
- [实验索引 E001–E146](docs/EXPERIMENTS.md) · [解码证据总表](docs/DECODE_MAP.md) · [当前证据边界](docs/CURRENT_STATUS.md)
- [公开版复现与数据来源](docs/PUBLIC_EDITION.md) · [来源与哈希清单](provenance/public_snapshot.json) · [Changelog](CHANGELOG.md)

## Quick start

Git LFS 提供论文 PDF、Qin 参考矩阵和三份校验矩阵。安装 Git LFS 后：

```sh
git clone https://github.com/spacelibres/StarlinkDownlinkAnalysis.git
cd StarlinkDownlinkAnalysis
git lfs pull
python tools/verify_public.py
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_code28_corner_constraints.py
python tools/run_experiment.py --list
```

| 目录 | 内容 |
|---|---|
| `experiments/`, `reports/` | 方法、脚本、阶段报告和中期论文 |
| `runs/` | 原始运行清单的公开字段、来源哈希和三份校验矩阵 |
| `learning/` | 公式索引、Notebook 和合成编码示例 |
| `data/`, `provenance/`, `references/` | 输入下载来源、哈希、授权参考实现和说明 |
| `tools/`, `tests/` | 复现入口、核验脚本和单元测试 |

录波按需从发行方下载。部分实验依赖本地录波或外部工程证据，相关要求见复现说明。公开版版本号和本地原始档案版本号独立。

