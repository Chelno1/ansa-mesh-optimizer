"""
报告生成模块

提供优化器比较和优化过程的报告生成功能
"""

from .comparison_reporter import ComparisonReporter
from .optimization_reporter import OptimizationReporter

__all__ = ["ComparisonReporter", "OptimizationReporter"]
