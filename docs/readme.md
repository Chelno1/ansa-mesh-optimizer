# ANSA Mesh Optimizer (增强版本)

一个用于ANSA有限元网格参数优化的高级工具集，支持多种优化算法和智能化参数调优。

![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 项目简介

ANSA Mesh Optimizer 是一个专门为ANSA有限元分析软件设计的网格参数优化工具。通过集成多种先进的优化算法，自动搜索最优的网格参数组合，以最小化不合格网格单元数量，提高网格质量和分析精度。

### 🎯 主要目标

- **自动化优化**: 自动搜索最优网格参数，减少手动调参时间
- **多算法支持**: 提供贝叶斯优化、遗传算法、随机搜索等多种优化策略
- **智能缓存**: 避免重复计算，提高优化效率
- **可视化分析**: 生成详细的优化报告和可视化图表
- **易于集成**: 支持命令行和Python API两种使用方式

## ✨ 核心特性

### 🔧 优化算法
- **贝叶斯优化** - 基于高斯过程的智能搜索
- **遗传算法** - 具有自适应变异和多样性保持
- **随机森林优化** - 基于决策树的优化策略
- **随机搜索** - 简单快速的基线方法
- **并行优化** - 多进程并行参数搜索

### 🛠️ 智能功能
- **早停机制** - 自动检测收敛，避免过度优化
- **参数验证** - 确保参数在合理范围内
- **结果缓存** - 智能缓存避免重复计算
- **敏感性分析** - 分析参数对结果的影响程度
- **内存优化** - 高效的内存管理和垃圾回收

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
python main.py info --check-deps
```

## 🚀 快速开始

### 1. 基本优化
```bash
# 使用贝叶斯优化（推荐）
python main.py optimize --optimizer bayesian --n-calls 30 --evaluator mock

# 使用遗传算法
python main.py optimize --optimizer genetic --n-calls 50 --evaluator mock
```

### 2. 优化器比较
```bash
# 比较多个优化器性能
python main.py compare --optimizers bayesian random genetic --n-calls 20 --n-runs 3
```

### 3. 使用真实ANSA环境
```bash
# 确保ANSA环境可用
python main.py info --check-ansa

# 运行真实优化
python main.py optimize --optimizer bayesian --evaluator ansa --config my_config.json
```

## 📖 详细使用指南

### 命令行界面

#### 主要命令

| 命令 | 描述 | 示例 |
|------|------|------|
| `optimize` | 运行单个优化器 | `python main.py optimize --optimizer bayesian` |
| `compare` | 比较多个优化器 | `python main.py compare --optimizers bayesian genetic` |
| `config` | 配置管理 | `python main.py config generate` |
| `info` | 系统信息 | `python main.py info --check-deps` |
| `test` | 运行测试 | `python main.py test --quick` |

#### optimize 命令参数

```bash
python main.py optimize [OPTIONS]

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
python main.py compare [OPTIONS]

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
from ansa_mesh_optimizer_improved import optimize_mesh_parameters, MeshOptimizer
from compare_optimizers_improved import compare_optimizers

# 单次优化
result = optimize_mesh_parameters(
    n_calls=30,
    optimizer='bayesian',
    evaluator_type='mock',
    use_cache=True
)

print(f"最佳参数: {result['best_params']}")
print(f"最佳值: {result['best_value']:.6f}")

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
from config import config_manager
from mesh_evaluator import create_mesh_evaluator

# 自定义配置
config_manager.optimization_config.n_calls = 50
config_manager.optimization_config.early_stopping = True
config_manager.optimization_config.patience = 10

# 创建优化器实例
optimizer = MeshOptimizer(
    config=config_manager.optimization_config,
    evaluator_type='ansa',
    use_cache=True
)

# 执行优化
result = optimizer.optimize(optimizer='bayesian')

# 敏感性分析
sensitivity = optimizer.sensitivity_analysis(
    best_params=result['best_params'],
    n_trials=5
)

# 保存结果
optimizer.save_best_params('best_params.txt')
```
## 🎨 中文字体配置
本项目完美支持中文图表显示，自动检测系统字体：
自动配置
python# 字体会自动配置，无需手动设置
from font_config import test_chinese_display

### 测试中文显示效果
test_chinese_display()
使用装饰器
pythonfrom font_decorator import with_chinese_font

@with_chinese_font
def my_plot_function():
    plt.title("中文标题")
    plt.xlabel("X轴标签")
    plt.ylabel("Y轴标签")
    plt.show()
手动安装字体（如需要）
bash# 运行字体诊断
python font_diagnosis.py

### 自动安装中文字体
python install_chinese_fonts.py

## ⚙️ 配置管理

### 生成默认配置

```bash
# 生成默认配置文件
python main.py config generate --output default_config.json

# 生成示例配置文件
python main.py config generate --output example_config.json --example
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
python main.py config validate my_config.json

# 显示当前配置
python main.py config show

# 显示特定配置节
python main.py config show --section optimization
```

## 📁 项目结构

```
ansa-mesh-optimizer/
├── 📁 core/                          # 核心模块
│   ├── ansa_mesh_optimizer_improved.py    # 主优化器
│   ├── genetic_optimizer_improved.py      # 遗传算法
│   ├── compare_optimizers_improved.py     # 优化器比较
│   └── early_stopping.py                  # 早停机制
├── 📁 config/                         # 配置管理
│   ├── config.py                          # 配置模块
│   └── default_config.json               # 默认配置
├── 📁 evaluators/                     # 评估器
│   ├── mesh_evaluator.py                 # 网格评估器
│   └── batch_mesh_improved.py            # 批处理脚本
├── 📁 utils/                          # 工具模块
│   ├── utils.py                          # 通用工具
│   ├── optimization_cache.py             # 缓存管理
│   ├── font_config.py                    # 字体配置
│   └── font_decorator.py                 # 字体装饰器
├── 📁 tests/                          # 测试脚本
│   ├── test_decorator.py                 # 装饰器测试
│   ├── font_diagnosis.py                 # 字体诊断
│   └── fix_test.py                       # 修复验证
├── 📄 main.py                         # 主程序入口
├── 📄 requirements.txt                # 项目依赖
├── 📄 README.md                       # 项目说明
└── 📄 CHANGELOG.md                    # 更新日志
```

## 🔧 模块详解

### MeshOptimizer (主优化器)
- 支持多种优化算法
- 集成缓存和早停机制
- 自动生成优化报告
- 参数敏感性分析

### 优化算法模块
- **贝叶斯优化**: 基于scikit-optimize的高效搜索
- **遗传算法**: 自适应参数和多样性保持
- **随机搜索**: 快速基线方法
- **并行优化**: 多进程参数搜索

### 评估器模块
- **ANSA评估器**: 真实ANSA环境集成
- **Mock评估器**: 测试和开发用模拟器
- **多种测试函数**: Rosenbrock, Ackley, Rastrigin等

### 缓存系统
- **文件缓存**: pickle格式持久化存储
- **数据库缓存**: SQLite数据库存储
- **智能清理**: 自动清理过期缓存
- **统计信息**: 命中率和性能监控

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
python main.py optimize --optimizer parallel --n-calls 100

# 并行比较
python main.py compare --parallel-runs --optimizers bayesian genetic
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
python main.py test

# 快速测试
python main.py test --quick

# 字体功能测试
python test_decorator.py

# 完整测试
python main.py test --evaluator mock --verbose-test

# 性能测试
python main.py info --performance
```

### 系统检查

```bash
# 检查依赖库
python main.py info --check-deps

# 检查ANSA环境
python main.py info --check-ansa

# 完整系统信息
python main.py info --check-deps --check-ansa --performance
```

## 📈 示例和用例

### 示例1: 基本优化工作流程

```python
# 1. 设置配置
from config import config_manager

config_manager.optimization_config.n_calls = 30
config_manager.optimization_config.use_cache = True

# 2. 运行优化
from ansa_mesh_optimizer_improved import optimize_mesh_parameters

result = optimize_mesh_parameters(
    optimizer='bayesian',
    evaluator_type='mock'
)

# 3. 分析结果
print(f"最优参数: {result['best_params']}")
print(f"目标值: {result['best_value']:.6f}")
print(f"执行时间: {result['execution_time']:.2f}秒")
```

### 示例2: 优化器性能比较

```python
from compare_optimizers_improved import compare_optimizers

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

### 示例3: 自定义遗传算法

```python
from genetic_optimizer_improved import GeneticOptimizer, GeneticConfig
from mesh_evaluator import create_mesh_evaluator
from config import config_manager

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
   python main.py info --check-ansa
   
   # 使用模拟评估器
   python main.py optimize --evaluator mock
   ```

2. **缺少依赖库**
   ```bash
   # 检查依赖
   python main.py info --check-deps
   
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
    python main.py info --check-deps

    # 安装缺失依赖
    pip install scikit-optimize
    ```

7. **参数验证失败**

    ```bash
    # 检查参数配置
    python main.py config validate config.json

    # 生成默认配置
    python main.py config generate
    ```

### 日志和调试

```bash
# 启用详细日志
python main.py optimize --verbose --log-file optimization.log

# 保存详细报告
python main.py optimize --save-plots --output results.json
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
# 在genetic_optimizer_improved.py中实现新算法
class NewOptimizer:
    def __init__(self, param_space, evaluator, config):
        self.param_space = param_space
        self.evaluator = evaluator
        self.config = config
    
    def optimize(self, n_calls, **kwargs):
        # 实现优化逻辑
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimizer_name': 'New Optimizer'
        }
```

#### 添加新的评估器

```python
# 在mesh_evaluator.py中添加新评估器
class CustomEvaluator(MeshEvaluator):
    def evaluate_mesh(self, params):
        # 实现评估逻辑
        return float(quality_score)
    
    def validate_params(self, params):
        # 实现参数验证
        return True
```

## 📄 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。

## 👥 作者和贡献者

- **Chel** - 主要开发者

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

---

**注意**: 本工具仅用于学术研究和工程应用，使用前请确保遵守相关软件许可协议。