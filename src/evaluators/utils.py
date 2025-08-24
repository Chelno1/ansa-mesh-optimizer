#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
网格评估器工具函数模块

此模块包含参数标准化等通用工具函数。
I/O 相关功能已移动到 io_utils.py 模块。

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def normalize_params(params: Dict[str, Any]) -> Dict[str, float]:
    """
    标准化参数字典，确保类型正确

    Args:
        params: 参数字典

    Returns:
        标准化后的参数字典
    """
    normalized = {}

    for key, value in params.items():
        if hasattr(value, "item"):  # numpy类型
            normalized[key] = float(value.item())
        elif hasattr(value, "dtype"):  # numpy数组等
            if hasattr(value, "size") and value.size == 1:
                normalized[key] = float(value.item())
            else:
                normalized[key] = (
                    float(value.tolist()[0])
                    if hasattr(value, "tolist")
                    else float(value)
                )
        else:
            normalized[key] = float(value)

    return normalized
