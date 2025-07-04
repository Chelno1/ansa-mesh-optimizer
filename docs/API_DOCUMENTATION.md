# ANSA 网格优化器 API 文档

## 概述

ANSA 网格优化器是一个高性能的网格优化工具，提供多种优化算法和配置选项。本文档详细介绍了项目的 API 接口和使用方法。

## 版本信息

- **版本**: 1.3.0
- **作者**: Chel
- **更新日期**: 2025-07-04

## 核心模块

### 1. 配置管理 (`src.config`)

#### UnifiedConfigManager

统一配置管理器，负责管理所有配置选项。

```python
from src.config.config_refactored import UnifiedConfigManager

# 创建配置管理器
config_manager = UnifiedConfigManager()

# 从文件加载配置
config_manager = UnifiedConfigManager('config.json')

# 保存配置
config_manager.save_config('output_config.json')
```

**主要方法:**

- `validate_all_configs()`: 验证所有配置
- `load_config(config_file: str)`: 从文件加载配置
- `save_config(config_file: str)`: 保存配置到文件
- `get_config_summary()`: 获取配置摘要

#### OptimizationConfig

优化配置类，包含优化算法相关的所有参数。

```python
from src.config.config_refactored import OptimizationConfig, OptimizerType

config = OptimizationConfig(
    n_calls=50,
    optimizer=OptimizerType.BAYESIAN,
    early_stopping=True,
    patience=10
)
```

**主要属性:**

- `n_calls`: 优化调用次数
- `optimizer`: 优化器类型
- `early_stopping`: 是否启用早停
- `patience`: 早停耐心值
- `use_cache`: 是否使用缓存

#### AnsaConfig

ANSA 软件配置类，包含 ANSA 相关的所有设置。

```python
from src.config.config_refactored import AnsaConfig
from pathlib import Path

config = AnsaConfig(
    ansa_executable='ansa',
    input_model='model.ansa',
    output_dir=Path('output'),
    execution_timeout=600
)
```

**主要属性:**

- `ansa_executable`: ANSA 可执行文件路径
- `input_model`: 输入模型文件
- `output_dir`: 输出目录
- `execution_timeout`: 执行超时时间

#### UnifiedParameterSpace

统一参数空间定义，消除了参数重复问题。

```python
from src.config.config_refactored import UnifiedParameterSpace

param_space = UnifiedParameterSpace()

# 获取参数名称
param_names = param_space.get_parameter_names()

# 获取参数边界
bounds = param_space.get_bounds()

# 获取 ANSA 映射
mapping = param_space.get_ansa_mapping()
```

**主要方法:**

- `get_parameter(name: str)`: 获取特定参数定义
- `get_parameter_names()`: 获取所有参数名称
- `get_bounds()`: 获取参数边界
- `get_ansa_mapping()`: 获取 ANSA 参数映射
- `validate_bounds()`: 验证参数边界
- `to_skopt_space()`: 转换为 scikit-optimize 空间

### 2. 核心优化器 (`src.core`)

#### AnsaMeshOptimizer

主要的网格优化器类。

```python
from src.core.ansa_mesh_optimizer_improved import AnsaMeshOptimizer

optimizer = AnsaMeshOptimizer(config_manager)
result = optimizer.optimize()
```

**主要方法:**

- `optimize()`: 执行优化
- `evaluate_parameters(params)`: 评估参数
- `save_results(filename)`: 保存结果

### 3. 依赖管理 (`src.utils.dependency_manager`)

#### DependencyManager

统一依赖管理器，处理可选依赖的导入和检查。

```python
from src.utils.dependency_manager import dependency_manager, require, get_optional

# 获取必需依赖
numpy = require('numpy')

# 获取可选依赖
matplotlib = get_optional('matplotlib')

# 检查依赖是否可用
if dependency_manager.is_available('scikit-optimize'):
    # 使用贝叶斯优化
    pass
```

**主要功能:**

- 自动检测可用依赖
- 提供统一的导入接口
- 支持依赖替代方案
- 生成依赖报告

### 4. 异常处理 (`src.utils.exceptions`)

#### 自定义异常类

项目定义了多个专用异常类，提供更好的错误处理。

```python
from src.utils.exceptions import (
    ConfigurationError,
    OptimizationError,
    ValidationError,
    handle_exceptions
)

# 使用异常处理装饰器
@handle_exceptions()
def my_function():
    # 函数实现
    pass
```

**异常类型:**

- `AnsaMeshOptimizerError`: 基础异常类
- `ConfigurationError`: 配置错误
- `OptimizationError`: 优化错误
- `ValidationError`: 验证错误
- `FileOperationError`: 文件操作错误
- `DependencyError`: 依赖错误

## 使用示例

### 基本使用

```python
from src.config.config_refactored import UnifiedConfigManager
from src.core.ansa_mesh_optimizer_improved import AnsaMeshOptimizer

# 1. 创建配置管理器
config_manager = UnifiedConfigManager()

# 2. 自定义配置
config_manager.optimization_config.n_calls = 100
config_manager.optimization_config.optimizer = OptimizerType.BAYESIAN

# 3. 创建优化器
optimizer = AnsaMeshOptimizer(config_manager)

# 4. 执行优化
result = optimizer.optimize()

# 5. 保存结果
optimizer.save_results('optimization_results.json')
```

### 配置文件使用

```python
# 1. 创建示例配置文件
config_manager = UnifiedConfigManager()
config_manager.save_config('example_config.json')

# 2. 从配置文件加载
config_manager = UnifiedConfigManager('example_config.json')

# 3. 修改配置
config_manager.optimization_config.n_calls = 200

# 4. 保存修改后的配置
config_manager.save_config('modified_config.json')
```

### 参数空间自定义

```python
from src.config.config_refactored import ParameterDefinition, ParameterType

# 获取参数空间
param_space = config_manager.parameter_space

# 修改参数边界
element_size_param = param_space.get_parameter('element_size')
if element_size_param:
    element_size_param.bounds = (0.3, 3.0)

# 验证参数空间
param_space.validate_bounds()
```

## 错误处理

### 配置验证

```python
try:
    config_manager.validate_all_configs()
except ConfigurationError as e:
    print(f"配置错误: {e}")
    print(f"错误代码: {e.error_code}")
    print(f"详细信息: {e.details}")
```

### 优化错误处理

```python
try:
    result = optimizer.optimize()
except OptimizationError as e:
    print(f"优化错误: {e}")
    if e.optimizer_type:
        print(f"优化器类型: {e.optimizer_type}")
    if e.iteration:
        print(f"错误发生在第 {e.iteration} 次迭代")
```

## 性能优化建议

### 1. 缓存使用

```python
# 启用缓存以避免重复计算
config_manager.optimization_config.use_cache = True
config_manager.optimization_config.cache_file = 'optimization_cache.pkl'
```

### 2. 并行处理

```python
# 设置并行作业数
config_manager.optimization_config.n_jobs = 4
```

### 3. 早停机制

```python
# 启用早停以避免过度优化
config_manager.optimization_config.early_stopping = True
config_manager.optimization_config.patience = 10
config_manager.optimization_config.min_delta = 0.01
```

## 扩展开发

### 添加新的优化器

1. 在 `OptimizerType` 枚举中添加新类型
2. 在优化器工厂中实现新的优化器类
3. 更新依赖管理器以检查所需依赖

### 添加新的参数

1. 在 `UnifiedParameterSpace` 中定义新参数
2. 添加相应的 ANSA 映射
3. 更新验证逻辑

### 自定义评估器

```python
from src.evaluators.mesh_evaluator import MeshEvaluator

class CustomEvaluator(MeshEvaluator):
    def evaluate(self, parameters):
        # 自定义评估逻辑
        return score
```

## 故障排除

### 常见问题

1. **依赖缺失**: 使用 `dependency_manager.print_dependency_report()` 检查依赖状态
2. **配置错误**: 使用 `validate_all_configs()` 验证配置
3. **参数边界错误**: 使用 `validate_bounds()` 检查参数边界

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 启用详细输出
config_manager.optimization_config.verbose = True
```

## 更新日志

### v1.3.0 (2025-07-04)

- 重构配置系统，消除参数重复
- 添加统一依赖管理
- 实现自定义异常处理
- 改进文件组织结构
- 添加完整的测试套件

### v1.2.0 (2025-06-20)

- 修复参数命名一致性
- 增强配置验证
- 改进错误处理

### v1.1.0 (2025-06-09)

- 添加多种优化算法
- 实现缓存机制
- 添加早停功能