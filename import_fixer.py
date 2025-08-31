#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局导入修复器
解决PyInstaller环境中的所有相对导入问题

作者: Chel
创建日期: 2025-08-31
"""

import sys
import os
import importlib.util
from importlib.machinery import ModuleSpec
from importlib.abc import MetaPathFinder, Loader
from typing import Optional


class PyInstallerImportFixer(MetaPathFinder):
    """PyInstaller环境下的导入修复器"""
    
    def __init__(self, base_path: str, src_path: str):
        self.base_path = base_path
        self.src_path = src_path
        
    def find_spec(self, fullname: str, path: Optional[list] = None, target: Optional[object] = None) -> Optional[ModuleSpec]:
        """查找模块规格"""
        
        # 处理相对导入的绝对路径转换
        if fullname.startswith('src.'):
            # 已经是绝对路径，直接处理
            module_path = self._find_module_file(fullname)
        elif '.' in fullname and not fullname.startswith('_'):
            # 可能是需要转换的相对导入
            parts = fullname.split('.')
            if len(parts) >= 2:
                # 尝试将其转换为src.*格式
                src_fullname = f"src.{fullname}"
                module_path = self._find_module_file(src_fullname)
                if module_path:
                    fullname = src_fullname
                else:
                    module_path = self._find_module_file(fullname)
            else:
                module_path = self._find_module_file(fullname)
        else:
            module_path = self._find_module_file(fullname)
            
        if module_path and os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location(fullname, module_path)
            return spec
            
        return None
        
    def _find_module_file(self, fullname: str) -> Optional[str]:
        """查找模块文件的实际路径"""
        # 替换点为路径分隔符
        module_path = fullname.replace('.', os.sep)
        
        # 尝试多个可能的位置
        candidates = [
            os.path.join(self.src_path, module_path + '.py'),
            os.path.join(self.base_path, module_path + '.py'),
            os.path.join(self.src_path, module_path, '__init__.py'),
            os.path.join(self.base_path, module_path, '__init__.py'),
        ]
        
        # 如果是src.开头的，也尝试去掉src前缀
        if fullname.startswith('src.'):
            stripped_name = fullname[4:]  # 去掉 'src.'
            stripped_path = stripped_name.replace('.', os.sep)
            candidates.extend([
                os.path.join(self.src_path, stripped_path + '.py'),
                os.path.join(self.src_path, stripped_path, '__init__.py'),
            ])
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
                
        return None


def setup_import_fixer(base_path: str, src_path: str):
    """设置导入修复器"""
    fixer = PyInstallerImportFixer(base_path, src_path)
    
    # 将修复器插入到sys.meta_path的开头
    if fixer not in sys.meta_path:
        sys.meta_path.insert(0, fixer)
        print(f"已安装导入修复器: base={base_path}, src={src_path}")


def patch_relative_imports():
    """修补所有已知的相对导入问题"""
    
    # 检查是否在PyInstaller环境中
    if not hasattr(sys, '_MEIPASS'):
        return
        
    print("开始修补相对导入...")
    
    # 常见的相对导入映射
    import_mappings = {
        # evaluators模块内的相对导入
        'evaluators.ansa_runner': 'src.evaluators.ansa_runner',
        'evaluators.temp_files': 'src.evaluators.temp_files', 
        'evaluators.validator': 'src.evaluators.validator',
        'evaluators.environment': 'src.evaluators.environment',
        'evaluators.parameter_replacement_strategies': 'src.evaluators.parameter_replacement_strategies',
        'evaluators.utils': 'src.evaluators.utils',
        'evaluators.io_utils': 'src.evaluators.io_utils',
        
        # cli模块内的相对导入
        'cli.commands.command_dispatcher': 'src.cli.commands.command_dispatcher',
        'cli.commands.optimize_cmd': 'src.cli.commands.optimize_cmd',
        'cli.commands.compare_cmd': 'src.cli.commands.compare_cmd',
        'cli.commands.config_cmd': 'src.cli.commands.config_cmd',
        'cli.commands.info_cmd': 'src.cli.commands.info_cmd',
        'cli.commands.test_cmd': 'src.cli.commands.test_cmd',
        
        # 其他常见模块
        'utils.logging_config': 'src.utils.logging_config',
        'utils.display_config': 'src.utils.display_config',
        'config.config': 'src.config.config',
        'core.ansa_mesh_optimizer': 'src.core.ansa_mesh_optimizer',
    }
    
    # 预加载关键模块
    for short_name, full_name in import_mappings.items():
        try:
            # 尝试导入完整模块名
            module = importlib.import_module(full_name)
            # 将其也注册为短名称
            sys.modules[short_name] = module
            print(f"  映射: {short_name} -> {full_name}")
        except ImportError as e:
            print(f"  跳过: {short_name} ({e})")
            continue
    
    print("相对导入修补完成")


def fix_module_paths():
    """修复模块路径问题"""
    if not hasattr(sys, '_MEIPASS'):
        return
        
    base_path = sys._MEIPASS
    src_path = os.path.join(base_path, 'src')
    
    # 确保src目录在Python路径中
    paths_to_add = [src_path, base_path]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"添加路径: {path}")
    
    # 设置导入修复器
    setup_import_fixer(base_path, src_path)
    
    # 修补相对导入
    patch_relative_imports()