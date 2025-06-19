# Ansa网格优化器

一个用于优化有限元网格参数的Python工具，支持多种优化算法并与Ansa软件集成。

## 主要特性

- 🚀 **多种优化算法**: 支持贝叶斯优化、随机搜索、森林优化、遗传算法等
- 🎯 **智能缓存**: 避免重复计算，提高优化效率
- ⏰ **早停机制**: 自动检测收敛，节省计算时间
- 📊 **可视化分析**: 丰富的图表和统计分析
- 🔧 **灵活配置**: 支持配置文件和命令行参数
- 📈 **敏感性分析**: 分析参数对结果的影响
- 🏆 **性能比较**: 自动比较不同优化器的性能

## 安装

### 基础安装

```bash
# 克隆项目
git clone <repository_url>
cd ansa-mesh-optimizer

# 安装依赖
pip install -r requirements.txt
```

### 开发环境安装

```bash
# 创建虚拟环境
python -m venv ansa_optimizer_env
source ansa_optimizer_env/bin/activate  # Linux/Mac
# 或
ansa_optimizer_env\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装开发工具（可选）
pip install pytest black flake8 mypy
```

## 快速开始

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
print(f"最佳值: {result['best_value']}")
```

### 2. 命令行使用

```bash
# 运行单个优化器
python main.py optimize --optimizer bayesian --n-calls 30

# 比较多个优化器
python main.py compare --optimizers bayesian random genetic --n-calls 20

# 使用配置文件
python main.py optimize --config config.json

# 查看帮助
python main.py --help
```

### 3. 优化器比较

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

## 配置文件

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