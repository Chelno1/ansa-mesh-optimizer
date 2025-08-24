#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ansa批处理模块包

作者: Chel
创建日期: 2025-08-24
版本: 2.0.0
功能: 模块化的Ansa批处理网格处理
"""

from .config import AnsaBatchConfig, create_default_config, load_config_from_file
from .runner import AnsaBatchMeshRunner, run_batch_mesh, check_element_quality_simple
from .report import (
    QualityReportGenerator,
    ResultAnalyzer,
    generate_quality_report,
    analyze_quality_results,
)

__version__ = "2.0.0"
__author__ = "Chel"

__all__ = [
    # 配置相关
    "AnsaBatchConfig",
    "create_default_config", 
    "load_config_from_file",
    
    # 运行器相关
    "AnsaBatchMeshRunner",
    "run_batch_mesh",
    "check_element_quality_simple",
    
    # 报告相关
    "QualityReportGenerator",
    "ResultAnalyzer", 
    "generate_quality_report",
    "analyze_quality_results",
]