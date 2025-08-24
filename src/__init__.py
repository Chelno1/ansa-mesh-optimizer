#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANSA 网格优化器主包

统一版本管理模块，所有其他模块应该从这里导入版本号。

作者: Chel
创建日期: 2025-08-24
"""

# 统一版本号定义
__version__ = "2.3.0"
__author__ = "Chel"
__email__ = "chel.china@gmail.com"
__description__ = "高级网格参数优化工具"

# 保持向后兼容的别名
APP_VERSION = __version__
APP_NAME = "Ansa Mesh Optimizer"

# 包级别导出
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "APP_VERSION",
    "APP_NAME",
]
