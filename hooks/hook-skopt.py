#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyInstaller钩子 - scikit-optimize依赖处理
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# 收集所有skopt子模块
hiddenimports = collect_submodules('skopt')

# 添加特定的隐藏导入
hiddenimports += [
    'skopt.space.space',
    'skopt.utils',
    'skopt.acquisition',
    'skopt.learning',
    'skopt.learning.gaussian_process',
    'skopt.learning.forest',
    'skopt.optimizer',
    'skopt.optimizer.base',
    'skopt.optimizer.gp',
    'skopt.optimizer.forest',
    'skopt.optimizer.gbrt',
    'skopt.optimizer.dummy',
    'skopt.callbacks',
    'skopt.plots',
    'skopt.sampler',
    'skopt.benchmarks',
]

# 收集数据文件
datas = collect_data_files('skopt')