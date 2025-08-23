#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils 包 - 统一导出接口

提供序列化、格式化、参数验证等工具函数的统一访问点。
保持向后兼容性，同时支持模块化的代码组织。

作者: Chel
创建日期: 2025-08-23
版本: 2.0.0
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
    create_progress_bar
)

# 从原始 utils 模块导入剩余的函数
from .utils import (
    normalize_params,
    validate_param_types,
    safe_divide,
    check_memory_usage,
    setup_numpy_print_options,
    performance_monitor,
    create_progress_callback,
    retry_on_exception,
    calculate_statistics,
    filter_dict_by_keys,
    extract_numbers_from_text,
    PATTERNS
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
    
    # 常量
    'PATTERNS'
]

# 版本信息
__version__ = '2.0.0'
__author__ = 'Chel'

# 模块说明
__doc_modules__ = {
    'serialization': '序列化和配置文件处理',
    'formatting': '时间、表格、数字格式化',
    'utils': '通用工具函数和性能监控'
}