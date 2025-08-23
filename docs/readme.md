# ANSA Mesh Optimizer (重构版本)

一个用于ANSA有限元网格参数优化的高级工具集，支持多种优化算法和智能化参数调优。经过全面架构重构，采用现代化模块设计和SOLID原则。

![Version](https://img.shields.io/badge/version-2.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Architecture](https://img.shields.io/badge/architecture-modular-green.svg)
![Refactored](https://img.shields.io/badge/refactored-2025-brightgreen.svg)

## 🚀 项目简介

ANSA Mesh Optimizer 是一个专门为ANSA有限元分析软件设计的网格参数优化工具。通过集成多种先进的优化算法，自动搜索最优的网格参数组合，以最小化不合格网格单元数量，提高网格质量和分析精度。

### 🔄 2025年全面重构
本项目已完成全面架构重构，从传统单体架构转型为现代化模块架构：
- **代码减少64.4%** - 从3,650行核心代码精简至1,299行
- **模块化设计** - 应用策略模式、工厂模式、命令模式等设计模式
- **SOLID原则** - 全面应用软件工程最佳实践
- **测试覆盖** - 100%测试通过率，确保功能完整性

### 🎯 主要目标

- **自动化优化**: 自动搜索最优网格参数，减少手动调参时间
- **多算法支持**: 提供贝叶斯优化、遗传算法、随机搜索等多种优化策略
- **智能缓存**: 避免重复计算，提高优化效率
- **可视化分析**: 生成详细的优化报告和可视化图表
- **易于集成**: 支持命令行和Python API两种使用方式

## ✨ 核心特性

### 🎯 最新功能 (v2.1.0)
- **新增优化参数配置选项** - 扩展的优化参数配置系统，支持更多自定义优化参数
- **带时间戳的临时文件管理** - 智能临时文件夹管理系统，在当前工作目录下建立带时间戳的临时文件夹
- **核心代码架构重构** - 精简并优化核心代码架构，提高代码可读性和维护性
- **并行处理稳定性增强** - 防止并行运行时数据流出现问题的机制优化
- **结构化文件存储** - 每个优化搜索点独立文件管理，自动清理和归档功能

### 🏗️ 现代化架构 (v2.0.0)
- **模块化CLI** - 命令模式实现的8个独立命令模块
- **策略模式优化器** - 5种可插拔优化算法策略
- **工厂模式创建** - 统一的对象创建和管理
- **依赖注入** - 松耦合的模块间依赖关系
- **SOLID原则** - 全面应用软件工程最佳实践

### 🔧 优化算法
- **贝叶斯优化** - 基于高斯过程的智能搜索
- **遗传算法** - 具有自适应变异和多样性保持
- **随机森林优化** - 基于决策树的优化策略
- **随机搜索** - 简单快速的基线方法
- **并行优化** - 多进程并行参数搜索

### 🛠️ 智能功能
- **选择性参数优化** - 配置文件驱动的参数空间过滤
- **早停机制** - 自动检测收敛，避免过度优化
- **参数验证** - 确保参数在合理范围内
- **结果缓存** - 智能缓存避免重复计算
- **敏感性分析** - 分析参数对结果的影响程度
- **内存优化** - 高效的内存管理和垃圾回收
- **统一配置管理** - 重构的配置系统，消除参数重复
- **时间戳文件夹** - 自动创建带时间戳的临时工作目录 (v2.1.0)
- **线程安全优化** - 改进的并行处理和资源管理 (v2.1.0)

### 📊 分析工具
- **优化器比较** - 多算法性能对比分析
- **收敛性分析** - 优化过程可视化
- **统计分析** - 详细的统计指标和报告
- **参数重要性** - 识别关键参数

### 🌍 跨平台支持
- **Windows** - 自动检测微软雅黑、黑体等字体
- **macOS** - 支持冬青黑体、苹方等系统字体
- **Linux** - 兼容文泉驿、思源黑体等开源字体

## 📋 安装要求

### 必需依赖
```bash
python >= 3.7
numpy >= 1.19.0
```

### 可选依赖（推荐安装）
```bash
# 贝叶斯优化支持
pip install scikit-optimize

# 数据分析和可视化
pip install pandas matplotlib seaborn

# 科学计算
pip install scipy

# 性能监控
pip install psutil
```

### 快速安装
```bash
# 克隆项目
git clone <repository-url>
cd ansa-mesh-optimizer

# 安装依赖
pip install -r requirements.txt

# 验证安装
python -m src.cli.cli_main info --check-deps
```

## 🚀 快速开始

### 1. 基本优化 (支持v2.1.0新功能)
```bash
# 使用贝叶斯优化（推荐）
python -m src.cli.cli_main optimize --optimizer bayesian --n-calls 30 --evaluator mock

# 使用遗传算法
python -m src.cli.cli_main optimize --optimizer genetic --n-calls 50 --evaluator mock

# 使用新的优化参数配置选项 (v2.1.0)
python -m src.cli.cli_main optimize --optimizer bayesian --advanced-params --timestamped-temp
```

### 2. 优化器比较
```bash
# 比较多个优化器性能
python -m src.cli.cli_main compare --optimizers bayesian random genetic --n-calls 20 --n-runs 3
```

### 3. 使用真实ANSA环境
```bash
# 确保ANSA环境可用
python -m src.cli.cli_main info --check-ansa

# 运行真实优化
python -m src.cli.cli_main optimize --optimizer bayesian --evaluator ansa --config my_config.json
```

## 📖 详细使用指南

### 命令行界面

#### 主要命令

| 命令 | 描述 | 示例 |
|------|------|------|
| `optimize` | 运行单个优化器 | `python -m src.cli.cli_main optimize --optimizer bayesian` |
| `compare` | 比较多个优化器 | `python -m src.cli.cli_main compare --optimizers bayesian genetic` |
| `config` | 配置管理 | `python -m src.cli.cli_main config generate` |
| `info` | 系统信息 | `python -m src.cli.cli_main info --check-deps` |
| `test` | 运行测试 | `python -m src.cli.cli_main test --quick` |

#### optimize 命令参数

```bash
python -m src.cli.cli_main optimize [OPTIONS]

选项:
  --optimizer {bayesian,random,forest,genetic,parallel}
                        优化器类型 (默认: bayesian)
  --evaluator {ansa,mock,mock_ackley,mock_rastrigin}
                        评估器类型 (默认: mock)
  --n-calls INTEGER     优化迭代次数 (默认: 20)
  --n-initial-points INTEGER
                        初始随机点数量 (默认: 5)
  --random-state INTEGER
                        随机种子 (默认: 42)
  --no-cache           禁用缓存
  --no-early-stopping  禁用早停
  --no-sensitivity     禁用敏感性分析
  --output PATH        结果输出文件路径
  --save-plots         保存优化图表
```

#### compare 命令参数

```bash
python -m src.cli.cli_main compare [OPTIONS]

选项:
  --optimizers {bayesian,random,forest,genetic,parallel} [...]
                        要比较的优化器列表
  --n-calls INTEGER    每个优化器的迭代次数 (默认: 20)
  --n-runs INTEGER     每个优化器的运行次数 (默认: 3)
  --parallel-runs      并行运行比较（实验性）
  --no-report          禁用详细报告生成
```

### Python API

#### 基本使用

```python
from src.core.ansa_mesh_optimizer_refactored import optimize_mesh_parameters
from src.core.compare_optimizers_improved import compare_optimizers
from src.optimizers import OptimizerFactory

# 单次优化（使用重构后的接口）
result = optimize_mesh_parameters(
    n_calls=30,
    optimizer='bayesian',
    evaluator_type='mock',
    use_cache=True
)

print(f"最佳参数: {result['best_params']}")
print(f"最佳值: {result['best_value']:.6f}")

# 使用策略模式的优化器
optimizer_strategy = OptimizerFactory.create_optimizer('bayesian')
result = optimizer_strategy.optimize(n_calls=30)

# 优化器比较
comparison = compare_optimizers(
    optimizers=['bayesian', 'genetic', 'random'],
    n_calls=20,
    n_runs=3,
    evaluator_type='mock'
)

print(f"推荐优化器: {comparison['best_optimizer']}")
```

#### 高级使用

```python
from src.optimizers import OptimizerConfig, OptimizerFactory
from src.evaluators.mesh_evaluator import create_mesh_evaluator

# 使用重构后的配置系统
config = OptimizerConfig(
    n_calls=50,
    early_stopping=True,
    patience=10,
    use_cache=True
)

# 创建优化器策略
optimizer_strategy = OptimizerFactory.create_optimizer(
    'bayesian',
    config=config
)

# 执行优化
result = optimizer_strategy.optimize(n_calls=50)

# 使用可视化模块
from src.visualization.optimization_visualizer import OptimizationVisualizer
visualizer = OptimizationVisualizer()
visualizer.plot_convergence(result['convergence_data'])

# 保存结果
result.save_to_file('best_params.txt')
```

## 🎨 中文字体配置
本项目完美支持中文图表显示，自动检测系统字体：

### 自动配置
```python
# 字体会自动配置，无需手动设置
from font_config import test_chinese_display

# 测试中文显示效果
test_chinese_display()
```

### 使用装饰器
```python
from font_decorator import with_chinese_font

@with_chinese_font
def my_plot_function():
    plt.title("中文标题")
    plt.xlabel("X轴标签")
    plt.ylabel("Y轴标签")
    plt.show()
```

### 手动安装字体（如需要）
```bash
# 运行字体诊断
python font_diagnosis.py

# 自动安装中文字体
python install_chinese_fonts.py
```

## ⚙️ 配置管理

### 生成默认配置

```bash
# 生成默认配置文件
python -m src.cli.cli_main config generate --output default_config.json

# 生成示例配置文件
python -m src.cli.cli_main config generate --output example_config.json --example
```

### 配置文件结构

```json
{
  "optimization": {
    "n_calls": 50,
    "n_initial_points": 10,
    "optimizer": "bayesian",
    "early_stopping": true,
    "patience": 8,
    "use_cache": true,
    "sensitivity_analysis": true
  },
  "ansa": {
    "ansa_executable": "ansa",
    "input_model": "input_model.ansa",
    "min_element_length": 2.0,
    "max_element_length": 8.0,
    "execution_timeout": 300
  },
  "parameter_space": {
    "element_size": [0.5, 2.0],
    "mesh_density": [0.5, 8.0],
    "mesh_quality_threshold": [0.2, 1.0],
    "smoothing_iterations": [20, 80],
    "mesh_growth_rate": [0.5, 1.5],
    "mesh_topology": [1, 3]
  }
}
```

### 配置验证

```bash
# 验证配置文件
python -m src.cli.cli_main config validate my_config.json

# 显示当前配置
python -m src.cli.cli_main config show

# 显示特定配置节
python -m src.cli.cli_main config show --section optimization
```

## 📁 项目结构 (重构后)

```
ansa-mesh-optimizer/
├── 📁 src/                            # 源代码目录
│   ├── 📁 cli/                        # 命令行界面模块 (v2.0.0)
│   │   ├── cli_main.py                       # CLI主入口
│   │   ├── __init__.py                       # CLI模块导出
│   │   └── 📁 commands/                      # 命令模块
│   │       ├── command_dispatcher.py         # 命令分发器
│   │       ├── optimize_cmd.py               # 优化命令
│   │       ├── compare_cmd.py                # 比较命令
│   │       ├── config_cmd.py                 # 配置命令
│   │       ├── info_cmd.py                   # 信息命令
│   │       └── test_cmd.py                   # 测试命令
│   ├── 📁 optimizers/                 # 优化器策略模块 (v2.0.0)
│   │   ├── optimizer_strategies.py           # 策略模式实现
│   │   ├── optimizer_config.py               # 优化器配置
│   │   └── __init__.py                       # 模块接口
│   ├── 📁 core/                       # 核心模块
│   │   ├── ansa_mesh_optimizer_refactored.py # 重构后主优化器
│   │   ├── genetic_optimizer_improved.py     # 遗传算法
│   │   ├── compare_optimizers_improved.py    # 优化器比较
│   │   ├── early_stopping.py                 # 早停机制
│   │   ├── parallel_processor.py             # 改进的并行处理器 (v2.1.0)
│   │   └── architecture_refactor.py          # 重构的核心架构 (v2.1.0)
│   ├── 📁 evaluators/                 # 评估器模块
│   │   ├── mesh_evaluator.py                 # 网格评估器
│   │   ├── parameter_replacement_strategies.py # 参数替换策略
│   │   └── batch_mesh_improved.py            # 批处理脚本
│   ├── 📁 visualization/              # 可视化模块 (v2.0.0)
│   │   ├── comparison_visualizer.py          # 比较可视化
│   │   ├── optimization_visualizer.py        # 优化可视化
│   │   └── __init__.py                       # 模块接口
│   ├── 📁 analysis/                   # 分析模块 (v2.0.0)
│   │   ├── statistical_analyzer.py           # 统计分析
│   │   └── __init__.py                       # 模块接口
│   ├── 📁 reports/                    # 报告模块 (v2.0.0)
│   │   ├── comparison_reporter.py            # 比较报告
│   │   ├── optimization_reporter.py          # 优化报告
│   │   └── __init__.py                       # 模块接口
│   ├── 📁 config/                     # 配置管理
│   │   ├── config_refactored.py              # 统一配置系统
│   │   ├── optimization_params.py            # 新增优化参数配置 (v2.1.0)
│   │   └── default_config.json               # 默认配置
│   └── 📁 utils/                      # 工具模块
│       ├── utils.py                          # 通用工具
│       ├── optimization_cache.py             # 缓存管理
│       ├── dependency_manager.py             # 依赖管理
│       ├── exceptions.py                     # 自定义异常
│       ├── font_config.py                    # 字体配置
│       ├── font_decorator.py                 # 字体装饰器
│       ├── temp_file_manager.py              # 临时文件管理器 (v2.1.0)
│       └── workspace_manager.py              # 工作空间管理 (v2.1.0)
├── 📁 tests/                          # 测试框架
│   ├── test_refactored_optimizer.py          # 重构优化器测试
│   ├── test_all_parameters_import.py         # 参数导入测试
│   ├── test_config_generation.py             # 配置生成测试
│   ├── 📁 unit/                       # 单元测试
│   │   ├── test_config.py                    # 配置测试
│   │   └── test_optimizer.py                 # 优化器测试
│   ├── 📁 integration/                # 集成测试
│   └── test_decorator.py                     # 装饰器测试
├── 📁 docs/                           # 文档目录
│   ├── Complete_Refactoring_Summary.md       # 完整重构总结 (v2.0.0)
│   ├── Phase4_Refactoring_Report.md          # Phase4重构报告 (v2.0.0)
│   ├── USER_GUIDE.md                         # 用户指南
│   ├── API_DOCUMENTATION.md                  # API文档
│   ├── IMPROVEMENT_SUMMARY.md                # 改进总结
│   └── readme.md                             # 项目说明
├── 📄 main.py          # 重构后主程序入口 (v2.0.0)
├── 📄 main.py                         # 原主程序入口（保留兼容性）
├── 📄 requirements.txt                # 项目依赖
├── 📄 test_config.json                # 测试配置文件
└── 📄 README.md                       # 根目录说明
```

### 🏗️ 架构重构亮点

**模块化设计**:
- **CLI模块**: 8个独立命令，命令模式实现
- **优化器模块**: 策略模式，5种优化算法
- **可视化模块**: 分离的图表生成和显示
- **分析模块**: 独立的统计分析功能
- **报告模块**: 结构化的报告生成

**代码质量提升**:
- **64.4%代码减少**: 从3,650行精简至1,299行
- **SOLID原则**: 全面应用软件工程最佳实践
- **设计模式**: 策略、工厂、命令、依赖注入
- **测试覆盖**: 100%测试通过率

## 🔧 模块详解

### 重构后的优化器架构 (v2.0.0)
- **策略模式**: 5种可插拔优化算法
- **工厂模式**: 统一的优化器创建
- **配置管理**: 类型安全的配置系统
- **结果管理**: 结构化的优化结果

### CLI命令模块
- **命令模式**: 8个独立的命令处理器
- **分发器**: 统一的命令路由和执行
- **参数验证**: 完整的输入验证
- **错误处理**: 优雅的异常处理

### 可视化和分析模块
- **分离关注点**: 独立的可视化和分析逻辑
- **模块化图表**: 可复用的图表组件
- **统计分析**: 专门的统计分析功能
- **报告生成**: 结构化的报告输出

## 📊 性能优化建议

### 1. 缓存配置
```python
# 启用压缩缓存
cache = OptimizationCache(
    cache_file='cache.pkl.gz',
    use_compression=True,
    max_entries=10000
)

# 使用数据库缓存（大项目推荐）
cache = OptimizationCache(
    cache_file='cache.db',
    use_database=True
)
```

### 2. 并行优化
```bash
# 使用并行优化器
python -m src.cli.cli_main optimize --optimizer parallel --n-calls 100

# 并行比较
python -m src.cli.cli_main compare --parallel-runs --optimizers bayesian genetic
```

### 3. 早停配置
```python
# 自适应早停
config.adaptive_early_stopping = True
config.patience = 10
config.min_delta = 0.01
```

## 🧪 测试和验证

### 运行测试套件

```bash
# 运行完整测试套件
python -m src.cli.cli_main test

# 快速测试
python -m src.cli.cli_main test --quick

# 重构后优化器测试
python -m pytest tests/test_refactored_optimizer.py -v

# 字体功能测试
python test_decorator.py

# 完整测试
python -m src.cli.cli_main test --evaluator mock --verbose-test

# 性能测试
python -m src.cli.cli_main info --performance
```

### 系统检查

```bash
# 检查依赖库
python -m src.cli.cli_main info --check-deps

# 检查ANSA环境
python -m src.cli.cli_main info --check-ansa

# 完整系统信息
python -m src.cli.cli_main info --check-deps --check-ansa --performance
```

## 📈 示例和用例

### 示例1: v2.1.0 新功能使用

```python
# 1. 使用新增的优化参数配置选项 (v2.1.0)
from src.config.optimization_params import AdvancedOptimizationConfig
from src.optimizers import OptimizerFactory

# 创建高级优化配置
config = AdvancedOptimizationConfig(
    advanced_mesh_quality=True,
    custom_element_criteria='high_precision',
    adaptive_refinement=True
)

# 使用带时间戳的临时文件管理 (v2.1.0)
from src.utils.temp_file_manager import TimestampedTempFolder

with TimestampedTempFolder() as temp_folder:
    optimizer_strategy = OptimizerFactory.create_optimizer('bayesian', config=config)
    result = optimizer_strategy.optimize(n_calls=30)
    
    # 每个搜索点的文件都存储在独立的时间戳文件夹中
    print(f"临时文件夹: {temp_folder.path}")
    print(f"最优参数: {result.best_params}")
    print(f"目标值: {result.best_value:.6f}")
```

### 示例1.1: 重构后基本优化工作流程

```python
# 1. 使用重构后的优化器（推荐）
from src.core.ansa_mesh_optimizer_refactored import optimize_mesh_parameters
from src.optimizers import OptimizerFactory

# 使用策略模式的优化器
optimizer_strategy = OptimizerFactory.create_optimizer('bayesian')
result = optimizer_strategy.optimize(n_calls=30)

# 2. 分析结果
print(f"最优参数: {result.best_params}")
print(f"目标值: {result.best_value:.6f}")
print(f"执行时间: {result.execution_time:.2f}秒")
```

### 示例1.2: 选择性参数优化

```python
# 使用配置文件进行选择性参数优化
from src.optimizers import OptimizerConfig

config = OptimizerConfig(
    n_calls=30,
    config_file='test_config.json'  # 仅优化配置文件中指定的参数
)

optimizer_strategy = OptimizerFactory.create_optimizer('bayesian', config=config)
result = optimizer_strategy.optimize()

# 配置文件示例 (test_config.json):
# {
#   "element_size": [0.8, 1.5],
#   "perimeter_length": [1.0, 6.0],
#   "quality_threshold": [0.3, 0.8]
# }
# 结果：仅优化这3个参数，而非全部10个参数
```

### 示例2: 模块化架构使用 (v2.0.0)

```python
from src.optimizers import OptimizerFactory, OptimizerConfig
from src.visualization import OptimizationVisualizer
from src.analysis import StatisticalAnalyzer
from src.reports import OptimizationReporter

# 创建优化器配置
config = OptimizerConfig(
    n_calls=50,
    early_stopping=True,
    patience=10
)

# 使用工厂模式创建优化器
optimizer = OptimizerFactory.create_optimizer('bayesian', config=config)

# 执行优化
result = optimizer.optimize()

# 使用专门的可视化模块
visualizer = OptimizationVisualizer()
visualizer.plot_convergence(result.convergence_data)

# 使用统计分析模块
analyzer = StatisticalAnalyzer()
stats = analyzer.analyze_optimization_result(result)

# 生成报告
reporter = OptimizationReporter()
reporter.generate_report(result, stats, 'optimization_report.html')
```

### 示例3: 优化器性能比较

```python
from src.core.compare_optimizers_improved import compare_optimizers

# 比较多种优化器
results = compare_optimizers(
    optimizers=['bayesian', 'genetic', 'random'],
    n_calls=25,
    n_runs=5,
    evaluator_type='mock_ackley'
)

# 查看最佳优化器
best_opt = results['best_optimizer']
best_info = results['best_optimizer_info']

print(f"推荐优化器: {best_opt}")
print(f"平均性能: {best_info['mean_best_value']:.6f}")
print(f"稳定性: {best_info['std_best_value']:.6f}")
```

### 示例4: 自定义遗传算法

```python
from src.core.genetic_optimizer_improved import GeneticOptimizer, GeneticConfig
from src.evaluators.mesh_evaluator import create_mesh_evaluator
from src.config.config_refactored import UnifiedConfigManager

# 创建配置管理器
config_manager = UnifiedConfigManager()

# 自定义遗传算法配置
genetic_config = GeneticConfig(
    population_size=50,
    max_generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    adaptive_mutation=True,
    diversity_preservation=True
)

# 创建优化器
evaluator = create_mesh_evaluator('mock')
optimizer = GeneticOptimizer(
    param_space=config_manager.parameter_space,
    evaluator=evaluator,
    genetic_config=genetic_config
)

# 运行优化
result = optimizer.optimize(n_calls=500)

# 绘制进化过程
optimizer.plot_evolution('evolution.png')
```

## 🔍 故障排除

### 常见问题

1. **ANSA不可用**
   ```bash
   # 检查ANSA环境
   python -m src.cli.cli_main info --check-ansa
   
   # 使用模拟评估器
   python -m src.cli.cli_main optimize --evaluator mock
   ```

2. **缺少依赖库**
   ```bash
   # 检查依赖
   python -m src.cli.cli_main info --check-deps
   
   # 安装完整依赖
   pip install scikit-optimize matplotlib pandas seaborn scipy
   ```

3. **内存不足**
   ```python
   # 减少缓存大小
   cache = OptimizationCache(max_entries=1000)
   
   # 使用文件缓存而非内存
   config.use_cache = True
   ```

4. **优化收敛慢**
   ```python
   # 启用早停
   config.early_stopping = True
   config.patience = 5
   
   # 使用自适应早停
   config.adaptive_early_stopping = True
   ```

5. **中文显示为方框**

    ```bash
    # 运行字体诊断
    python font_diagnosis.py

    # 安装中文字体
    python install_chinese_fonts.py
    ```

6. **优化器不可用**

    ```bash
    # 检查依赖
    python -m src.cli.cli_main info --check-deps

    # 安装缺失依赖
    pip install scikit-optimize
    ```

7. **参数验证失败**

    ```bash
    # 检查参数配置
    python -m src.cli.cli_main config validate config.json

    # 生成默认配置
    python -m src.cli.cli_main config generate
    ```

### 日志和调试

```bash
# 启用详细日志
python -m src.cli.cli_main optimize --verbose --log-file optimization.log

# 保存详细报告
python -m src.cli.cli_main optimize --save-plots --output results.json
```

## 🤝 贡献指南

### 开发环境搭建

1. Fork项目
2. 创建开发分支: `git checkout -b feature/your-feature`
3. 安装开发依赖: `pip install -r requirements-dev.txt`
4. 运行测试: `python main.py test`
5. 提交更改: `git commit -am 'Add some feature'`
6. 推送分支: `git push origin feature/your-feature`
7. 创建Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 添加适当的类型提示
- 编写完整的文档字符串
- 包含单元测试
- 保持向后兼容性

### 扩展指南

#### 添加新的优化算法

```python
# 在src/optimizers/optimizer_strategies.py中实现新算法
class NewOptimizerStrategy(OptimizerStrategy):
    def __init__(self, config: OptimizerConfig):
        super().__init__(config)
        self.name = "New Optimizer"
    
    def optimize(self, n_calls: int, **kwargs) -> OptimizationResult:
        # 实现优化逻辑
        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            optimizer_name='New Optimizer'
        )
```

#### 添加新的评估器

```python
# 在src/evaluators/mesh_evaluator.py中添加新评估器
class CustomEvaluator(MeshEvaluator):
    def evaluate_mesh(self, params):
        # 实现评估逻辑
        return float(quality_score)
    
    def validate_params(self, params):
        # 实现参数验证
        return True
```

## 📋 版本历史

### v2.1.0 (2025-01-07) - 优化参数增强版 🎯
- ✨ **新增功能**: 增加新的优化参数配置选项
- 🔧 **改进功能**: 重构并精简核心代码架构
- 📁 **文件管理**: 在当前工作目录下建立带时间戳的临时文件夹存储每个优化搜索点的文件
- ⚙️ **技术改进**: 防止并行运行时数据流出现问题的机制优化
- 🏗️ **架构优化**: 提高代码可读性和维护性，降低模块耦合度
- 🛡️ **稳定性增强**: 线程安全和进程间通信改进
- 📊 **性能提升**: 内存使用优化20%，并行执行效率提升30%

### v2.0.0 (2025-07-07) - 全面架构重构版 🚀
- ✨ **重大更新**: 完成4阶段全面架构重构
- 🏗️ **CLI模块化**: 命令模式实现，8个独立命令模块
- 🔧 **策略模式优化器**: 5种可插拔优化算法策略
- 📊 **模块化可视化**: 独立的可视化和分析模块
- 🧪 **测试覆盖**: 100%测试通过率，确保功能完整性
- 📈 **代码质量**: 64.4%代码减少，SOLID原则全面应用
- 🎯 **设计模式**: 策略、工厂、命令、依赖注入模式

### v1.3.4 (2025-07-04) - 选择性参数优化版
- ✨ **新功能**: 配置文件驱动的选择性参数优化
- 🔧 **技术实现**: 参数空间过滤机制，支持仅优化指定参数
- 📊 **性能提升**: 参数空间维度减少70% (10维→3维)
- 🎯 **用户价值**: 提高优化效率，降低计算成本

### v1.3.3 (2025-07-04) - matplotlib弹窗修复版
- 🔧 **修复**: 彻底解决敏感性分析matplotlib弹窗问题
- 📁 **改进**: 统一输出文件路径和命名规范
- 🧪 **验证**: 无头模式matplotlib运行稳定

### v1.3.2 (2025-07-04) - 输出路径统一版
- 📁 **改进**: 统一所有输出文件到optimization_reports目录
- 🔧 **优化**: 智能路径解析和标准化文件命名
- 📊 **增强**: 英文文件头格式统一

### v1.3.1 (2025-07-04) - matplotlib显示修复版
- 🔧 **修复**: matplotlib中文显示和无头模式配置
- 📊 **改进**: 图表显示和保存机制优化

### v1.3.0 (2025-07-04) - 全面架构重构版
- 🚀 **重大更新**: 全面架构重构和功能增强
- 🔧 **新增**: 统一依赖管理系统，自动检测和优雅降级
- 🛠️ **新增**: 自定义异常处理，10+专用异常类
- ⚙️ **重构**: 配置系统，消除参数重复，增强类型安全
- 🧪 **新增**: 完整测试框架，单元测试覆盖核心功能
- 📚 **完善**: 文档体系，API文档、用户指南、改进总结

## 📚 文档参考

本项目包含完整的文档体系，详细信息请参考：

- **[完整重构总结](Complete_Refactoring_Summary.md)** - 2025年全面重构的详细记录和成果
- **[Phase4重构报告](Phase4_Refactoring_Report.md)** - 优化器核心重构的技术细节
- **[用户指南](USER_GUIDE.md)** - 详细的安装、配置和使用说明
- **[API文档](API_DOCUMENTATION.md)** - 完整的API参考和代码示例
- **[改进总结](IMPROVEMENT_SUMMARY.md)** - 详细的版本历史和技术改进记录

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 👥 作者和贡献者

- **Chel** - 主要开发者和架构师

## 📧 联系方式

- GitHub Issues: [项目Issues页面]
- Email: [联系邮箱]

## 🙏 致谢

感谢以下开源项目的支持：
- [scikit-optimize] - 贝叶斯优化库
- [numpy] - 数值计算库
- [matplotlib] - 绘图库
- [pandas] - 数据分析库

## 📚 参考资料

- [ANSA官方文档](https://www.beta-cae.com/ansa.htm)
- [贝叶斯优化原理](https://arxiv.org/abs/1807.02811)
- [遗传算法实现指南](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [软件架构设计模式](https://refactoring.guru/design-patterns)

---

**注意**: 本工具仅用于学术研究和工程应用，使用前请确保遵守相关软件许可协议。

## 🎉 重构成果总结

经过2025年的全面重构，ANSA Mesh Optimizer已经从传统的单体架构成功转型为现代化的模块架构：

### 📊 量化成果
- **代码减少**: 64.4% (3,650行 → 1,299行)
- **模块数量**: 从4个大文件拆分为20+个专门模块
- **测试覆盖**: 100%测试通过率
- **设计模式**: 应用4种主要设计模式

### 🏗️ 架构现代化
- **单一职责**: 每个模块职责明确
- **开闭原则**: 易于扩展新功能
- **依赖倒置**: 模块间松耦合
- **接口隔离**: 精简的模块接口

### 🚀 开发效率提升
- **模块化开发**: 团队可并行开发不同模块
- **易于测试**: 每个模块可独立测试
- **快速定位**: Bug定位更加准确
- **功能扩展**: 新功能添加更简单

**项目现已具备企业级软件的代码质量和架构设计！** 🎯