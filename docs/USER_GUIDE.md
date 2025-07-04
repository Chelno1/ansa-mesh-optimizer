# ANSA 网格优化器用户指南

## 目录

1. [快速开始](#快速开始)
2. [安装指南](#安装指南)
3. [基本概念](#基本概念)
4. [配置说明](#配置说明)
5. [使用教程](#使用教程)
6. [高级功能](#高级功能)
7. [故障排除](#故障排除)
8. [最佳实践](#最佳实践)

## 快速开始

### 5分钟快速体验

```bash
# 1. 克隆项目
git clone <repository-url>
cd ansa-mesh-optimizer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行基本优化
python src/main.py --input-model example.ansa --optimizer bayesian --n-calls 20
```

### 基本工作流程

1. **准备模型**: 准备 ANSA 模型文件
2. **配置参数**: 设置优化参数和约束
3. **选择算法**: 选择合适的优化算法
4. **执行优化**: 运行优化过程
5. **分析结果**: 查看和分析优化结果

## 安装指南

### 系统要求

- **操作系统**: Windows 10+, Linux, macOS
- **Python**: 3.8+
- **内存**: 建议 8GB+
- **ANSA**: 需要安装 ANSA 软件

### 依赖安装

#### 必需依赖

```bash
pip install numpy
```

#### 可选依赖

```bash
# 贝叶斯优化支持
pip install scikit-optimize

# 绘图和可视化
pip install matplotlib seaborn

# 数据分析
pip install pandas

# 系统监控
pip install psutil
```

#### 一键安装

```bash
pip install -r requirements.txt
```

### 验证安装

```python
from src.utils.dependency_manager import dependency_manager

# 打印依赖报告
dependency_manager.print_dependency_report()
```

## 基本概念

### 优化算法

#### 1. 贝叶斯优化 (Bayesian)
- **适用场景**: 评估成本高，需要高效搜索
- **优点**: 收敛快，全局搜索能力强
- **缺点**: 需要 scikit-optimize 依赖

#### 2. 随机搜索 (Random)
- **适用场景**: 快速探索，基准测试
- **优点**: 简单可靠，无额外依赖
- **缺点**: 收敛较慢

#### 3. 森林优化 (Forest)
- **适用场景**: 中等复杂度问题
- **优点**: 平衡探索和利用
- **缺点**: 需要 scikit-optimize 依赖

#### 4. 遗传算法 (Genetic)
- **适用场景**: 复杂多模态问题
- **优点**: 全局搜索，处理离散变量
- **缺点**: 参数调优复杂

### 参数类型

#### 连续参数
- **element_size**: 单元尺寸 (0.5-2.0 mm)
- **perimeter_length**: 周边长度 (0.5-8.0 mm)
- **quality_threshold**: 质量阈值 (0.2-1.0)

#### 离散参数
- **distortion_distance**: 扭曲距离 (10-30%)
- **smoothing_iterations**: 平滑迭代次数 (20-80)
- **mesh_topology**: 网格拓扑类型 (1-3)

## 配置说明

### 配置文件结构

```json
{
  "optimization": {
    "n_calls": 50,
    "optimizer": "bayesian",
    "early_stopping": true,
    "patience": 10,
    "use_cache": true
  },
  "ansa": {
    "ansa_executable": "ansa",
    "input_model": "model.ansa",
    "execution_timeout": 600,
    "quality_check_enabled": true
  },
  "parameters": {
    "element_size": {
      "bounds": [0.5, 2.0],
      "default_value": 1.0
    }
  }
}
```

### 关键配置项

#### 优化配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_calls` | int | 20 | 优化调用次数 |
| `optimizer` | str | "bayesian" | 优化算法 |
| `early_stopping` | bool | true | 是否启用早停 |
| `patience` | int | 5 | 早停耐心值 |
| `use_cache` | bool | true | 是否使用缓存 |

#### ANSA 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ansa_executable` | str | "ansa" | ANSA 可执行文件 |
| `execution_timeout` | int | 300 | 执行超时时间(秒) |
| `max_retries` | int | 3 | 最大重试次数 |
| `quality_check_enabled` | bool | true | 是否启用质量检查 |

## 使用教程

### 教程 1: 基本优化

```python
from src.config.config_refactored import UnifiedConfigManager
from src.core.ansa_mesh_optimizer_improved import AnsaMeshOptimizer

# 1. 创建配置
config = UnifiedConfigManager()

# 2. 设置基本参数
config.optimization_config.n_calls = 30
config.ansa_config.input_model = "my_model.ansa"

# 3. 创建优化器
optimizer = AnsaMeshOptimizer(config)

# 4. 执行优化
result = optimizer.optimize()

print(f"最优参数: {result.x}")
print(f"最优值: {result.fun}")
```

### 教程 2: 使用配置文件

```python
# 1. 创建配置文件
config = UnifiedConfigManager()
config.save_config("my_config.json")

# 2. 编辑配置文件 (使用文本编辑器)

# 3. 从配置文件加载
config = UnifiedConfigManager("my_config.json")

# 4. 运行优化
optimizer = AnsaMeshOptimizer(config)
result = optimizer.optimize()
```

### 教程 3: 命令行使用

```bash
# 基本使用
python src/main.py --input-model model.ansa

# 指定优化器
python src/main.py --input-model model.ansa --optimizer genetic

# 设置调用次数
python src/main.py --input-model model.ansa --n-calls 100

# 使用配置文件
python src/main.py --config config.json

# 启用详细输出
python src/main.py --input-model model.ansa --verbose

# 保存结果
python src/main.py --input-model model.ansa --output results.json
```

### 教程 4: 参数空间自定义

```python
# 获取参数空间
param_space = config.parameter_space

# 查看所有参数
print("可用参数:", param_space.get_parameter_names())

# 修改参数边界
element_param = param_space.get_parameter('element_size')
if element_param:
    element_param.bounds = (0.3, 3.0)  # 扩大搜索范围

# 验证修改
param_space.validate_bounds()
```

## 高级功能

### 并行优化

```python
# 启用并行处理
config.optimization_config.n_jobs = 4  # 使用4个CPU核心

# 注意：并行处理需要确保ANSA许可证足够
```

### 缓存机制

```python
# 启用缓存
config.optimization_config.use_cache = True
config.optimization_config.cache_file = "optimization_cache.pkl"

# 清除缓存
import os
if os.path.exists("optimization_cache.pkl"):
    os.remove("optimization_cache.pkl")
```

### 早停机制

```python
# 配置早停
config.optimization_config.early_stopping = True
config.optimization_config.patience = 10  # 10次迭代无改善则停止
config.optimization_config.min_delta = 0.01  # 最小改善阈值
```

### 敏感性分析

```python
# 启用敏感性分析
config.optimization_config.sensitivity_analysis = True
config.optimization_config.sensitivity_trials = 5
config.optimization_config.noise_level = 0.1
```

### 结果可视化

```python
import matplotlib.pyplot as plt

# 优化历史可视化
def plot_optimization_history(result):
    plt.figure(figsize=(10, 6))
    plt.plot(result.func_vals)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.title('优化历史')
    plt.grid(True)
    plt.show()

# 参数重要性分析
def plot_parameter_importance(result):
    # 实现参数重要性可视化
    pass
```

## 故障排除

### 常见问题

#### 1. ANSA 无法启动

**问题**: `FileNotFoundError: ansa command not found`

**解决方案**:
```python
# 指定ANSA完整路径
config.ansa_config.ansa_executable = "/path/to/ansa/bin/ansa"
```

#### 2. 依赖缺失

**问题**: `ImportError: No module named 'skopt'`

**解决方案**:
```bash
# 安装缺失依赖
pip install scikit-optimize

# 或使用不需要该依赖的优化器
config.optimization_config.optimizer = OptimizerType.RANDOM
```

#### 3. 内存不足

**问题**: 优化过程中内存耗尽

**解决方案**:
```python
# 减少并行作业数
config.optimization_config.n_jobs = 1

# 设置内存限制
config.ansa_config.max_memory_usage = 4.0  # 4GB

# 启用临时文件清理
config.ansa_config.temp_cleanup = True
```

#### 4. 优化收敛慢

**问题**: 优化过程收敛缓慢

**解决方案**:
```python
# 增加初始采样点
config.optimization_config.n_initial_points = 10

# 使用更高效的优化器
config.optimization_config.optimizer = OptimizerType.BAYESIAN

# 调整早停参数
config.optimization_config.patience = 15
```

### 调试技巧

#### 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 启用优化器详细输出
config.optimization_config.verbose = True
```

#### 检查依赖状态

```python
from src.utils.dependency_manager import dependency_manager

# 打印依赖报告
dependency_manager.print_dependency_report()

# 检查特定依赖
if dependency_manager.is_available('scikit-optimize'):
    print("贝叶斯优化可用")
else:
    print("贝叶斯优化不可用，将使用随机搜索")
```

#### 验证配置

```python
try:
    config.validate_all_configs()
    print("配置验证通过")
except Exception as e:
    print(f"配置错误: {e}")
```

## 最佳实践

### 1. 选择合适的优化器

- **小规模问题** (< 20 次调用): 使用随机搜索
- **中等规模问题** (20-100 次调用): 使用贝叶斯优化
- **大规模问题** (> 100 次调用): 使用遗传算法
- **多模态问题**: 使用遗传算法

### 2. 参数空间设计

```python
# 合理设置参数边界
# 过宽：搜索效率低
# 过窄：可能错过最优解

# 示例：根据经验设置合理边界
element_size_bounds = (0.5, 2.0)  # 基于材料特性
quality_threshold_bounds = (0.3, 0.9)  # 基于质量要求
```

### 3. 性能优化

```python
# 启用缓存避免重复计算
config.optimization_config.use_cache = True

# 合理设置并行数
import os
n_cores = os.cpu_count()
config.optimization_config.n_jobs = min(n_cores, 4)

# 使用早停避免过度优化
config.optimization_config.early_stopping = True
```

### 4. 结果验证

```python
# 保存详细结果
optimizer.save_results("detailed_results.json")

# 验证最优参数
best_params = result.x
validation_score = optimizer.evaluate_parameters(best_params)

# 多次运行验证稳定性
for i in range(3):
    result_i = optimizer.optimize()
    print(f"运行 {i+1}: {result_i.fun}")
```

### 5. 监控和日志

```python
# 设置进度回调
def progress_callback(result):
    print(f"迭代 {len(result.func_vals)}: 当前最优值 = {result.fun}")

# 在优化器中使用回调
# optimizer.optimize(callback=progress_callback)
```

## 扩展和定制

### 添加自定义评估器

```python
from src.evaluators.mesh_evaluator import MeshEvaluator

class CustomQualityEvaluator(MeshEvaluator):
    def evaluate(self, parameters):
        # 实现自定义质量评估逻辑
        quality_score = self.calculate_custom_quality(parameters)
        return quality_score
    
    def calculate_custom_quality(self, parameters):
        # 自定义质量计算
        pass
```

### 集成外部工具

```python
# 集成其他CAE软件
class ExternalSolverEvaluator(MeshEvaluator):
    def evaluate(self, parameters):
        # 调用外部求解器
        result = self.run_external_solver(parameters)
        return result
```

## 社区和支持

### 获取帮助

1. **文档**: 查看完整的 API 文档
2. **示例**: 参考 `examples/` 目录中的示例
3. **测试**: 运行 `tests/` 目录中的测试用例

### 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

### 版本更新

定期检查项目更新：

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

---

**注意**: 本指南基于 ANSA 网格优化器 v1.3.0。如有疑问，请参考最新的 API 文档或联系开发团队。