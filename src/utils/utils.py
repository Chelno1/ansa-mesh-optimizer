#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数验证工具模块 - 重构版本

专注于参数验证和类型转换功能。
其他功能已按职责分离到专门模块：
- 格式化功能: utils/formatting.py
- 序列化功能: utils/serialization.py
- 通用工具函数: utils/misc.py

作者: Chel
创建日期: 2025-06-19
版本: 3.0.0
更新日期: 2025-08-24
重构: 按领域拆分功能，遵循单一职责原则
"""

import logging
import time
from typing import Any, Dict, Union

import numpy as np

# 导入统一的参数验证功能
from .parameter_validator import normalize_params as _normalize_params
from .parameter_validator import validate_param_types as _validate_param_types

logger = logging.getLogger(__name__)


def normalize_params(params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    标准化参数字典，将numpy类型转换为Python原生类型
    使用统一的参数验证器

    Args:
        params: 参数字典

    Returns:
        标准化后的参数字典

    Examples:
        >>> params = {'quality_threshold': np.array([0.6]), 'value': np.float64(3.14)}
        >>> normalized = normalize_params(params)
        >>> print(normalized)
    """
    return _normalize_params(params)


def validate_param_types(
    params: Dict[str, Any], param_space
) -> Dict[str, Union[int, float]]:
    """
    验证并转换参数类型
    使用统一的参数验证器

    Args:
        params: 参数字典
        param_space: 参数空间定义

    Returns:
        验证后的参数字典

    Examples:
        >>> validated = validate_param_types(params, param_space_config)
        >>> print(validated)
    """
    return _validate_param_types(params, param_space)


if __name__ == "__main__":
    # 测试工具函数
    print("=== Utils Testing ===")

    # 测试参数标准化
    test_params = {
        "quality_threshold": np.array([0.6]),
        "distortion_distance": np.float64(20.0),
        "normal_param": 3.0,
    }

    normalized = normalize_params(test_params)
    print(f"Normalized params: {normalized}")

    # 导入misc模块的函数进行测试
    from .misc import (
        calculate_statistics,
        create_progress_callback,
        filter_dict_by_keys,
        performance_monitor,
    )

    # 测试性能监控
    with performance_monitor("Test operation"):
        time.sleep(0.1)

    # 测试进度回调
    progress_cb = create_progress_callback(10)
    for i in range(11):
        progress_cb(i, current_best=10 - i)
        time.sleep(0.05)

    # 测试统计计算
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = calculate_statistics(test_values)
    print(f"Statistics: {stats}")

    # 测试字典过滤
    test_dict = {"a": 1, "b": 2, "c": 3, "d": 4}
    filtered = filter_dict_by_keys(test_dict, ["a", "c"], include=True)
    print(f"Filtered dict: {filtered}")

    print("Utils testing completed!")
