# Ansa网格优化器

一个用于优化有限元网格参数的高级Python工具，支持多种优化算法并与Ansa软件无缝集成。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/your-username/ansa-mesh-optimizer)

## 🌟 主要特性

- 🚀 **多种优化算法**: 支持贝叶斯优化、随机搜索、森林优化、遗传算法等
- 🎯 **智能缓存系统**: 避免重复计算，显著提高优化效率
- ⏰ **早停机制**: 自动检测收敛，节省宝贵的计算时间
- 📊 **丰富的可视化**: 收敛图、参数相关性、统计分析等
- 🔧 **灵活配置管理**: 支持JSON配置文件和命令行参数
- 📈 **敏感性分析**: 深入分析参数对结果的影响程度
- 🏆 **性能比较**: 自动比较不同优化器的性能表现
- 🔄 **并行计算**: 支持多进程并行优化
- 🎨 **交互式界面**: 友好的命令行工具

## 📋 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [功能特性](#功能特性)
- [使用指南](#使用指南)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [高级功能](#高级功能)
- [故障排除](#故障排除)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## 🚀 安装

### 基础安装

```bash
# 克隆项目
git clone https://github.com/your-username/ansa-mesh-optimizer.git
cd ansa-mesh-optimizer

# 安装依赖
pip install -r requirements.txt
```

### 使用虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv ansa_optimizer_env

# 激活虚拟环境
source ansa_optimizer_env/bin/activate  # Linux/Mac
# 或
ansa_optimizer_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 最小安装

如果只需要核心功能：

```bash
pip install numpy scikit-optimize matplotlib pandas scipy deap
```

## ⚡ 快速开始

### 1. 基本优化

```python
from ansa_mesh_optimizer_improved import optimize_mesh_parameters

# 使用模拟评估器进行快速测试
result = optimize_mesh_parameters(
    n_calls=20,
    optimizer='bayesian',
    evaluator_type='mock'
)

print(f"最佳参数: {result['best_params']}")
print(f"最佳值: {result['best_value']:.6f}")
print(f"执行时间: {result['execution_time']:.2f}秒")
```

### 2. 命令行使用

```bash
# 运行贝叶斯优化
python main.py optimize --optimizer bayesian --n-calls 30 --evaluator mock

# 比较多个优化器
python main.py compare --optimizers bayesian random genetic --n-calls 20 --evaluator mock

# 使用真实Ansa评估器
python main.py optimize --optimizer genetic --n-calls 15 --evaluator ansa

# 查看详细帮助
python main.py optimize --help
```

### 3. 批量比较

```python
from compare_optimizers_improved import compare_optimizers

# 比较不同优化器的性能
results = compare_optimizers(
    optimizers=['bayesian', 'random', 'genetic'],
    n_calls=20,
    n_runs=3,
    evaluator_type='mock'
)

print(f"最佳优化器: {results['best_optimizer']}")
```

## 🔧 功能特性

### 支持的优化算法

| 优化器 | 描述 | 适用场景 | 推荐迭代次数 |
|--------|------|----------|-------------|
| `bayesian` | 贝叶斯优化（高斯过程） | 昂贵的目标函数，少量迭代 | 20-50 |
| `random` | 随机搜索 | 基准比较，快速探索 | 50-100 |
| `forest` | 森林优化（随机森林） | 中等复杂度问题 | 30-80 |
| `genetic` | 遗传算法 | 复杂多模态问题 | 50-200 |
| `parallel` | 并行随机搜索 | 多核处理器，快速评估 | 100-500 |

### 优化参数空间

默认优化的网格参数：

- **element_size**: 单元尺寸 (0.5 - 2.0)
- **mesh_density**: 网格密度 (1 - 5)
- **mesh_quality_threshold**: 质量阈值 (0.2 - 1.0)
- **smoothing_iterations**: 平滑迭代次数 (20 - 80)
- **mesh_growth_rate**: 网格增长率 (0.5 - 1.5)
- **mesh_topology**: 网格拓扑类型 (1 - 3)

## 📖 使用指南

### 配置文件

创建配置文件来自定义优化参数：

```bash
# 生成默认配置文件
python main.py config generate
```

配置文件示例：

```json
{
  "optimization": {
    "n_calls": 30,
    "optimizer": "bayesian",
    "early_stopping": true,
    "use_cache": true,
    "patience": 5,
    "min_delta": 0.01
  },
  "parameter_space": {
    "element_size": [0.5, 2.0],
    "mesh_density": [1, 5],
    "mesh_quality_threshold": [0.2, 1.0]
  },
  "ansa": {
    "min_element_length": 2.0,
    "max_element_length": 8.0
  }
}
```

### 与Ansa集成

#### 前提条件

1. 安装Ansa软件并确保Python可以导入ansa模块
2. 准备网格参数文件(.ansa_mpar)和质量标准文件(.ansa_qual)
3. 配置输入模型文件

#### 使用真实Ansa评估器

```python
# 在代码中使用
result = optimize_mesh_parameters(
    n_calls=20,
    optimizer='bayesian',
    evaluator_type='ansa'  # 使用真实Ansa评估器
)
```

```bash
# 命令行使用
python main.py optimize --optimizer genetic --evaluator ansa --n-calls 25
```

## 📊 结果分析

### 优化结果

优化完成后会自动生成：

- **收敛图**: 显示优化过程
- **参数相关性图**: 参数之间的关系
- **敏感性分析**: 参数影响程度
- **统计报告**: 详细的数值分析

### 结果文件结构

```
optimization_reports/
├── 20250619_142030_Bayesian_Optimization/
│   ├── optimization_report.txt      # 文本报告
│   ├── convergence.png               # 收敛图
│   ├── parameter_correlation.png     # 参数相关性
│   ├── early_stopping_history.png   # 早停历史
│   └── sensitivity_analysis.png     # 敏感性分析
└── best_params_Bayesian_Optimization_20250619_142030.txt
```

### 可视化示例

```python
# 查看优化历史
from ansa_mesh_optimizer_improved import MeshOptimizer

optimizer = MeshOptimizer(evaluator_type='mock')
result = optimizer.optimize(optimizer='bayesian', n_calls=30)

# 运行敏感性分析
sensitivity_results = optimizer.sensitivity_analysis(
    result['best_params'],
    n_trials=10,
    noise_level=0.1
)
```

## 🎯 高级功能

### 1. 并行优化

```python
# 使用多进程并行优化
result = optimize_mesh_parameters(
    optimizer='parallel',
    n_calls=100,
    n_workers=4  # 使用4个进程
)
```

### 2. 多目标优化

```python
from genetic_optimizer_improved import MultiObjectiveGeneticOptimizer

# 多目标优化（例如：最小化不合格网格和计算时间）
optimizer = MultiObjectiveGeneticOptimizer(
    param_space=param_space,
    evaluators=[mesh_evaluator, time_evaluator]
)

result = optimizer.optimize(n_calls=50)
pareto_front = result['pareto_front']
```

### 3. 自定义评估器

```python
from mesh_evaluator import MeshEvaluator

class CustomEvaluator(MeshEvaluator):
    def evaluate_mesh(self, params):
        # 自定义评估逻辑
        return custom_evaluation_function(params)
    
    def validate_params(self, params):
        # 参数验证逻辑
        return True

# 使用自定义评估器
optimizer = MeshOptimizer(evaluator=CustomEvaluator())
```

### 4. 缓存管理

```python
from optimization_cache import OptimizationCache

# 创建缓存实例
cache = OptimizationCache('my_optimization_cache.pkl')

# 查看缓存统计
stats = cache.get_stats()
print(f"缓存命中率: {stats['hit_rate']:.2%}")

# 导出缓存数据
cache.export_cache_data('cache_export.json')
```

### 5. 早停配置

```python
from config import config_manager

# 配置早停参数
config = config_manager.optimization_config
config.early_stopping = True
config.patience = 10
config.min_delta = 0.001
```

## 🔍 性能优化建议

### 1. 缓存配置

```python
# 启用缓存以避免重复计算
config_manager.optimization_config.use_cache = True
config_manager.optimization_config.cache_file = 'my_cache.pkl'
```

### 2. 早停配置

```python
# 配置早停以节省时间
config_manager.optimization_config.early_stopping = True
config_manager.optimization_config.patience = 10
config_manager.optimization_config.min_delta = 0.001
```

### 3. 参数空间调整

```python
# 基于经验缩小搜索空间
config_manager.parameter_space.element_size = (0.8, 1.2)  # 缩小范围
config_manager.parameter_space.mesh_density = (2, 4)      # 排除极值
```

### 4. 算法选择指南

- **快速测试**: 使用 `random` 优化器
- **精确优化**: 使用 `bayesian` 优化器
- **复杂问题**: 使用 `genetic` 优化器
- **大规模搜索**: 使用 `parallel` 优化器

## 📝 命令行参考

### 基本命令

```bash
# 查看帮助
python main.py --help
python main.py optimize --help
python main.py compare --help

# 显示系统信息
python main.py info
python main.py info --check-deps

# 配置管理
python main.py config generate
python main.py config validate config.json
```

### 优化命令

```bash
# 基本优化
python main.py optimize --optimizer bayesian --n-calls 20

# 高级选项
python main.py optimize \
    --optimizer genetic \
    --n-calls 50 \
    --evaluator ansa \
    --no-cache \
    --no-early-stopping \
    --output results.json \
    --config my_config.json
```

### 比较命令

```bash
# 基本比较
python main.py compare --optimizers bayesian random genetic

# 详细比较
python main.py compare \
    --optimizers bayesian random forest genetic \
    --n-calls 30 \
    --n-runs 5 \
    --evaluator mock \
    --no-sensitivity
```

## 🛠️ 故障排除

### 常见问题

#### 1. Ansa模块导入失败

```bash
# 错误信息
ModuleNotFoundError: No module named 'ansa'

# 解决方案
# 确保Ansa已正确安装并配置Python环境
# 或使用模拟评估器进行测试
python main.py optimize --evaluator mock
```

#### 2. 内存不足

```bash
# 减少并行进程数或缓存大小
python main.py optimize --optimizer bayesian --n-calls 10
```

#### 3. 优化结果不理想

```python
# 增加迭代次数或尝试不同优化器
result = optimize_mesh_parameters(
    n_calls=100,  # 增加迭代次数
    optimizer='genetic'  # 尝试遗传算法
)
```

#### 4. JSON序列化错误

这个问题已在v1.1.0中修复。如果仍遇到问题：

```bash
# 更新到最新版本
git pull origin main
pip install -r requirements.txt
```

### 调试模式

```bash
# 启用详细日志
python main.py optimize --verbose --log-file debug.log

# 检查依赖库
python main.py info --check-deps

# 运行诊断测试
python final_test.py
```

### 性能监控

```python
# 检查内存使用
from utils import check_memory_usage
memory_info = check_memory_usage()
print(f"内存使用: {memory_info['rss_mb']:.1f} MB")

# 查看缓存统计
cache_stats = optimizer.cache.get_stats()
print(f"缓存命中率: {cache_stats['hit_rate']:.2%}")
```

## 🧪 测试

### 运行测试

```bash
# 快速功能测试
python quick_fix_test.py

# 完整测试套件
python final_test.py

# CLI测试
python test_cli.py
```

### 单元测试

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行单元测试
pytest tests/ -v

# 测试覆盖率
pytest --cov=. tests/
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 开发流程

1. Fork本仓库
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 进行更改并添加测试
4. 确保所有测试通过: `python final_test.py`
5. 提交更改: `git commit -m 'Add amazing feature'`
6. 推送分支: `git push origin feature/amazing-feature`
7. 创建Pull Request

### 代码规范

- 遵循PEP 8风格指南
- 使用类型提示
- 为所有公共函数添加文档字符串
- 最大行长度: 88字符
- 运行 `black .` 进行代码格式化

### 测试要求

- 为新功能编写单元测试
- 保持测试覆盖率在80%以上
- 确保所有现有测试通过

## 📚 API文档

### 核心类

#### MeshOptimizer

```python
class MeshOptimizer:
    def __init__(self, config=None, evaluator_type='mock', use_cache=True):
        """初始化网格优化器"""
        
    def optimize(self, optimizer='bayesian', n_calls=20, **kwargs):
        """执行优化"""
        
    def sensitivity_analysis(self, best_params, n_trials=5, noise_level=0.1):
        """参数敏感性分析"""
```

#### MeshEvaluator

```python
class MeshEvaluator(ABC):
    @abstractmethod
    def evaluate_mesh(self, params: Dict[str, float]) -> float:
        """评估网格质量"""
        
    @abstractmethod
    def validate_params(self, params: Dict[str, float]) -> bool:
        """验证参数有效性"""
```

### 便捷函数

```python
def optimize_mesh_parameters(n_calls=20, optimizer='bayesian', 
                            evaluator_type='mock', **kwargs):
    """便捷的网格参数优化函数"""

def compare_optimizers(optimizers=['bayesian', 'random'], 
                      n_calls=20, n_runs=3, **kwargs):
    """比较多个优化器性能"""
```

## 📋 更新日志

### [1.1.0] - 2025-06-19

#### 新增功能
- ✨ 完整重构的代码架构
- ✨ 统一的配置管理系统
- ✨ 智能缓存机制
- ✨ 早停功能
- ✨ 多目标遗传算法(NSGA-II)
- ✨ 统计分析和高级可视化
- ✨ 命令行接口
- ✨ 性能比较工具
- ✨ 敏感性分析
- ✨ 并行优化支持

#### 改进
- 🔧 增强的错误处理和日志系统
- 🔧 改进的文档和示例
- 🔧 更好的类型注解
- 🔧 模块化设计

#### 修复
- 🐛 JSON序列化错误(numpy类型)
- 🐛 硬编码路径问题
- 🐛 缺少依赖检查
- 🐛 不充分的错误处理

#### 移除
- ❌ 旧版优化接口

### [1.0.0] - 2025-06-09

#### 新增功能
- ✨ 初始版本发布
- ✨ 基本优化功能
- ✨ Ansa集成
- ✨ 多种优化算法

## 🏆 致谢

感谢以下开源项目的贡献：

- [scikit-optimize](https://scikit-optimize.github.io/) - 贝叶斯优化算法
- [DEAP](https://deap.readthedocs.io/) - 遗传算法框架
- [matplotlib](https://matplotlib.org/) - 数据可视化
- [pandas](https://pandas.pydata.org/) - 数据分析
- [scipy](https://scipy.org/) - 科学计算
- [NumPy](https://numpy.org/) - 数值计算

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **作者**: Chel
- **邮箱**: [your-email@example.com]
- **项目主页**: [https://github.com/your-username/ansa-mesh-optimizer]
- **问题反馈**: [https://github.com/your-username/ansa-mesh-optimizer/issues]

## 🌟 支持项目

如果这个项目对您有帮助，请考虑：

- ⭐ 给项目加星
- 🐛 报告问题
- 💡 提出功能建议
- 🤝 贡献代码
- 📢 分享给同事

---

**让网格优化变得简单高效！** 🚀
# 生成默认配置文件
python main.py config generate
```

配置文件示例：

```json
{
  "optimization": {
    "n_calls": 30,
    "optimizer": "bayesian",
    "early_stopping": true,
    "use_cache": true,
    "patience": 5,
    "min_delta": 0.01
  },
  "parameter_space": {
    "element_size": [0.5, 2.0],
    "mesh_density": [1, 5],
    "mesh_quality_threshold": [0.2, 1.0]
  }
}
```

## 支持的优化器

| 优化器 | 描述 | 适用场景 |
|--------|------|----------|
| `bayesian` | 贝叶斯优化（高斯过程） | 昂贵的目标函数，少量迭代 |
| `random` | 随机搜索 | 基准比较，简单快速 |
| `forest` | 森林优化（随机森林） | 中等复杂度的问题 |
| `genetic` | 遗传算法 | 复杂的多模态问题 |
| `parallel` | 并行随机搜索 | 多核处理器，快速评估 |

## 参数空间

默认优化的网格参数：

- **element_size**: 单元尺寸 (0.5 - 2.0)
- **mesh_density**: 网格密度 (1 - 5)
- **mesh_quality_threshold**: 质量阈值 (0.2 - 1.0)
- **smoothing_iterations**: 平滑迭代次数 (20 - 80)
- **mesh_growth_rate**: 网格增长率 (0.5 - 1.5)
- **mesh_topology**: 网格拓扑类型 (1 - 3)

## 与Ansa集成

### 前提条件

1. 安装Ansa软件
2. 确保Python可以导入ansa模块
3. 准备好网格参数文件(.ansa_mpar)和质量标准文件(.ansa_qual)

### 使用真实Ansa评估器

```python
# 使用真实Ansa评估器
result = optimize_mesh_parameters(
    n_calls=20,
    optimizer='bayesian',
    evaluator_type='ansa'  # 使用真实Ansa评估器
)
```

```bash
# 命令行使用Ansa评估器
python main.py optimize --evaluator ansa --optimizer genetic
```

## 结果分析

### 优化结果

优化完成后会生成：

- **最佳参数**: 找到的最优参数组合
- **目标值**: 对应的不合格网格数量
- **收敛图**: 优化过程的可视化
- **参数相关性**: 参数之间的关系分析
- **敏感性分析**: 参数对结果的影响程度

### 结果文件

```
optimization_reports/
├── 20250619_142030_Bayesian_Optimization/
│   ├── optimization_report.txt
│   ├── convergence.png
│   ├── parameter_correlation.png
│   └── early_stopping_history.png
└── best_params_Bayesian_Optimization_20250619_142030.txt
```

## 高级功能

### 1. 并行优化

```python
# 使用多进程并行优化
result = optimize_mesh_parameters(
    optimizer='parallel',
    n_calls=100,
    n_workers=4  # 使用4个进程
)
```

### 2. 多目标优化

```python
from genetic_optimizer_improved import MultiObjectiveGeneticOptimizer

# 多目标优化（例如：最小化不合格网格数量和计算时间）
optimizer = MultiObjectiveGeneticOptimizer(
    param_space=param_space,
    evaluators=[mesh_evaluator, time_evaluator]
)

result = optimizer.optimize(n_calls=50)
pareto_front = result['pareto_front']
```

### 3. 自定义评估器

```python
from mesh_evaluator import MeshEvaluator

class CustomEvaluator(MeshEvaluator):
    def evaluate_mesh(self, params):
        # 自定义评估逻辑
        return custom_evaluation_function(params)
    
    def validate_params(self, params):
        # 参数验证逻辑
        return True

# 使用自定义评估器
optimizer = MeshOptimizer(evaluator=CustomEvaluator())
```

## 性能优化建议

### 1. 缓存配置

```python
# 启用缓存以避免重复计算
config_manager.optimization_config.use_cache = True
config_manager.optimization_config.cache_file = 'my_cache.pkl'
```

### 2. 早停配置

```python
# 配置早停以节省时间
config_manager.optimization_config.early_stopping = True
config_manager.optimization_config.patience = 10
config_manager.optimization_config.min_delta = 0.001
```

### 3. 参数空间调整

```python
# 基于经验缩小搜索空间
config_manager.parameter_space.element_size = (0.8, 1.2)  # 缩小范围
config_manager.parameter_space.mesh_density = (2, 4)      # 排除极值
```

## 故障排除

### 常见问题

1. **Ansa模块导入失败**
   ```
   解决方案: 确保Ansa已正确安装并配置Python环境
   ```

2. **内存不足**
   ```bash
   # 减少并行进程数或缓存大小
   python main.py optimize --optimizer bayesian --n-calls 10
   ```

3. **优化结果不理想**
   ```python
   # 增加迭代次数或尝试不同优化器
   result = optimize_mesh_parameters(
       n_calls=100,  # 增加迭代次数
       optimizer='genetic'  # 尝试遗传算法
   )
   ```

### 调试模式

```bash
# 启用详细日志
python main.py optimize --verbose --log-file debug.log

# 检查依赖库
python main.py info --check-deps
```

## 开发指南

### 代码结构

```
ansa-mesh-optimizer/
├── config.py                          # 配置管理
├── mesh_evaluator.py                  # 网格评估接口
├── optimization_cache.py              # 缓存管理
├── early_stopping.py                  # 早停机制
├── ansa_mesh_optimizer_improved.py    # 主优化器
├── genetic_optimizer_improved.py      # 遗传算法优化器
├── compare_optimizers_improved.py     # 优化器比较工具
├── batch_mesh_improved.py             # Ansa批处理脚本
├── main.py                            # 主程序入口
├── requirements.txt                   # 依赖库清单
└── README.md                          # 项目说明
```

### 运行测试

```bash
# 运行单元测试
pytest tests/

# 测试覆盖率
pytest --cov=. tests/

# 代码格式检查
flake8 .

# 类型检查
mypy .
```

### 贡献指南

1. Fork项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建Pull Request

## 版本历史

- **v1.1.0** (2025-06-19)
  - 完全重构代码架构
  - 添加配置管理系统
  - 实现缓存和早停机制
  - 增强可视化和统计分析
  - 改进错误处理和日志系统

- **v1.0.0** (2025-06-09)
  - 初始版本
  - 基本优化功能

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 作者: Chel
- 邮箱: [your-email@example.com]
- 项目主页: [项目URL]

## 致谢

感谢以下开源项目：

- [scikit-optimize](https://scikit-optimize.github.io/) - 贝叶斯优化算法
- [DEAP](https://deap.readthedocs.io/) - 遗传算法框架
- [matplotlib](https://matplotlib.org/) - 数据可视化
- [pandas](https://pandas.pydata.org/) - 数据分析

---

如果这个项目对您有帮助，请考虑给个⭐️！