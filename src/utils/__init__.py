#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils 包 - 统一导出接口

模块化重构后的工具函数包，按职责领域分离：
- utils.py: 参数验证和类型转换
- formatting.py: 时间、表格、数字格式化和文本处理
- serialization.py: JSON序列化和配置文件处理
- misc.py: 通用工具函数（数学计算、系统监控、性能分析等）

提供统一的导出接口，保持向后兼容性。

作者: Chel
创建日期: 2025-08-23
版本: 3.0.0
更新日期: 2025-08-24
重构: 按领域拆分功能，遵循单一职责原则
"""

# 从序列化模块导入
from .serialization import (
    safe_json_serialize,
    validate_file_path,
    load_json_config,
    save_json_config,
    parse_json_string,
    serialize_numpy_object
)

# 从格式化模块导入
from .formatting import (
    format_execution_time,
    create_summary_table,
    truncate_string,
    estimate_completion_time,
    create_backup_filename,
    format_number,
    format_percentage,
    format_file_size,
    create_progress_bar,
    extract_numbers_from_text,
    PATTERNS
)

# 从原始 utils 模块导入剩余的函数
from .utils import (
    normalize_params,
    validate_param_types
)

# 从 misc 模块导入通用工具函数
from .misc import (
    safe_divide,
    check_memory_usage,
    setup_numpy_print_options,
    performance_monitor,
    create_progress_callback,
    retry_on_exception,
    calculate_statistics,
    filter_dict_by_keys
)

# 从异常模块导入
from .exceptions import (
    AnsaMeshOptimizerError,
    ConfigurationError,
    ValidationError,
    OptimizationError,
    EvaluationError,
    FileOperationError,
    DependencyError,
    ResourceError,
    TimeoutError,
    ConvergenceError,
    ParameterError,
    handle_exceptions,
    ErrorCodes
)

# 从错误处理模块导入
from .error_handler import (
    safe_execute,
    handle_exceptions as handle_exceptions_decorator,
    validate_file_path as validate_file_path_error_handler,
    ensure_directory,
    log_function_call,
    retry_on_failure,
    ErrorContext,
    create_error_context,
    format_error_message,
    setup_global_error_handler
)

# 定义公开的API
__all__ = [
    # 序列化相关
    'safe_json_serialize',
    'validate_file_path',
    'load_json_config',
    'save_json_config',
    'parse_json_string',
    'serialize_numpy_object',
    
    # 格式化相关
    'format_execution_time',
    'create_summary_table',
    'truncate_string',
    'estimate_completion_time',
    'create_backup_filename',
    'format_number',
    'format_percentage',
    'format_file_size',
    'create_progress_bar',
    
    # 参数验证和处理
    'normalize_params',
    'validate_param_types',
    
    # 数学和统计
    'safe_divide',
    'calculate_statistics',
    
    # 性能监控
    'check_memory_usage',
    'performance_monitor',
    'create_progress_callback',
    
    # 工具函数
    'retry_on_exception',
    'filter_dict_by_keys',
    'extract_numbers_from_text',
    'setup_numpy_print_options',
    
    # 异常类
    'AnsaMeshOptimizerError',
    'ConfigurationError',
    'ValidationError',
    'OptimizationError',
    'EvaluationError',
    'FileOperationError',
    'DependencyError',
    'ResourceError',
    'TimeoutError',
    'ConvergenceError',
    'ParameterError',
    'ErrorCodes',
    
    # 错误处理工具
    'safe_execute',
    'handle_exceptions',
    'handle_exceptions_decorator',
    'validate_file_path_error_handler',
    'ensure_directory',
    'log_function_call',
    'retry_on_failure',
    'ErrorContext',
    'create_error_context',
    'format_error_message',
    'setup_global_error_handler',
    
    # 常量
    'PATTERNS'
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Chel'

# 模块说明
__doc_modules__ = {
    'utils': '参数验证和类型转换',
    'formatting': '时间、表格、数字格式化和文本处理',
    'serialization': '序列化和配置文件处理',
    'misc': '通用工具函数（数学计算、系统监控、性能分析等）',
    'exceptions': '自定义异常类定义',
    'error_handler': '错误处理装饰器和工具函数'
}
