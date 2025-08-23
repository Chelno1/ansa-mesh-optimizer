# 错误处理模块重构文档

## 概述

本次重构将原本臃肿的 `src/utils/error_handler.py` 文件进行了拆分和优化，提高了代码的可维护性和职责清晰度。

## 重构目标

1. **职责分离**: 将异常类定义与错误处理工具函数分离
2. **统一异常体系**: 使用现有的 `exceptions.py` 作为统一的异常定义模块
3. **提高复用性**: 异常类可以独立使用，不依赖错误处理工具
4. **保持兼容性**: 确保现有代码无需修改即可正常工作

## 重构后的模块结构

### 1. `src/utils/exceptions.py` - 异常类定义模块

**职责**: 专门存放所有自定义异常类的定义

**主要内容**:
- `AnsaMeshOptimizerError`: 基础异常类
- `ConfigurationError`: 配置错误
- `ValidationError`: 验证错误  
- `OptimizationError`: 优化过程错误
- `EvaluationError`: 评估过程错误
- `FileOperationError`: 文件操作错误
- `DependencyError`: 依赖错误
- `ResourceError`: 资源相关错误
- `TimeoutError`: 超时错误
- `ConvergenceError`: 收敛性错误
- `ParameterError`: 参数错误
- `ErrorCodes`: 错误代码常量类
- `handle_exceptions`: 基础异常处理装饰器

### 2. `src/utils/error_handler.py` - 错误处理工具模块

**职责**: 提供错误处理装饰器、工具函数和上下文管理器

**主要内容**:
- `safe_execute`: 安全执行函数
- `handle_exceptions`: 增强版异常处理装饰器
- `validate_file_path`: 文件路径验证
- `ensure_directory`: 确保目录存在
- `log_function_call`: 函数调用日志装饰器
- `retry_on_failure`: 失败重试装饰器
- `ErrorContext`: 错误上下文管理器
- `create_error_context`: 创建错误上下文
- `format_error_message`: 格式化错误消息
- `setup_global_error_handler`: 设置全局错误处理器

### 3. `src/utils/__init__.py` - 统一导出接口

**更新内容**:
- 添加了异常类的导出
- 添加了错误处理工具函数的导出
- 更新了模块文档说明

## 使用方式

### 1. 导入异常类

```python
# 直接从 exceptions 模块导入
from src.utils.exceptions import ConfigurationError, ValidationError

# 或者从统一的 utils 包导入
from src.utils import ConfigurationError, ValidationError
```

### 2. 导入错误处理工具

```python
# 直接从 error_handler 模块导入
from src.utils.error_handler import handle_exceptions, safe_execute

# 或者从统一的 utils 包导入
from src.utils import handle_exceptions, safe_execute
```

### 3. 使用示例

```python
from src.utils import ConfigurationError, handle_exceptions, safe_execute

# 使用异常类
def validate_config(config):
    if not config:
        raise ConfigurationError("配置不能为空")

# 使用装饰器
@handle_exceptions()
def risky_function():
    # 可能抛出异常的代码
    pass

# 使用安全执行
result = safe_execute(risky_function, default_return="默认值")
```

## 兼容性说明

- **向后兼容**: 现有代码无需修改，可以继续正常工作
- **导入路径**: 原有的导入路径仍然有效
- **功能保持**: 所有原有功能都得到保留

## 优势

1. **职责清晰**: 异常定义与错误处理工具分离
2. **复用性强**: 异常类可以独立使用
3. **维护性好**: 模块结构更清晰，便于维护
4. **扩展性强**: 可以更容易地添加新的异常类或工具函数

## 注意事项

1. 新的异常类应该添加到 `exceptions.py` 模块中
2. 新的错误处理工具应该添加到 `error_handler.py` 模块中
3. 更新 `__init__.py` 以导出新添加的功能
4. 保持异常类的继承关系和接口一致性

## 版本信息

- 重构版本: 2.0.0
- 重构日期: 2025-08-23
- 作者: Chel