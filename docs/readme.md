# ANSA Mesh Optimizer

高效的 ANSA 网格参数优化工具。

![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 项目简介

该工具旨在自动搜索 ANSA 网格生成中的最优参数组合，以减少不合格单元并提升网格质量。项目采用模块化结构，提供命令行接口和 Python API 两种使用方式。

## 核心特性

- **多种优化算法**：支持贝叶斯优化、随机搜索、随机森林和遗传算法，可在单机或并行模式下运行。
- **配置管理**：通过 `config` 模块定义参数空间和优化设置，并提供示例配置文件。
- **结果缓存**：内置缓存机制，避免重复计算并支持早停策略。
- **可扩展评估器**：在 `evaluators` 模块中自定义网格质量评估逻辑。
- **命令行工具**：在 `src/cli` 中提供 `info`、`config`、`optimize`、`compare` 等命令，便于集成与测试。
- **可视化与报告**：`reports` 和 `visualization` 模块生成收敛曲线和优化报告。

## 安装

```bash
pip install ansa-mesh-optimizer
```

或在源代码目录下进行本地安装：

```bash
git clone <repository-url>
cd ansa-mesh-optimizer
pip install -r docs/requirements.txt
```

## 快速开始

```bash
# 查看环境依赖
python -m src.cli.cli_main info --check-deps

# 运行一次优化（贝叶斯优化示例）
python -m src.cli.cli_main optimize --optimizer bayesian --n-calls 20 --evaluator mock

# 比较不同优化器
python -m src.cli.cli_main compare --optimizers bayesian genetic random
```

## 版本历史

### 0.2.0
- 初始公开版本，包含多种优化算法、配置系统、缓存机制及命令行工具。

## 许可证

本项目采用 [MIT 许可证](../LICENSE)。

