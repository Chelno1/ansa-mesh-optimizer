#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一依赖管理模块

作者: Chel
创建日期: 2025-07-04
版本: 1.0.0
功能: 统一管理项目的可选依赖，提供一致的导入接口
"""

import importlib
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DependencyInfo:
    """依赖信息类"""
    name: str
    import_name: str
    description: str
    required: bool = False
    version_check: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)

class DependencyManager:
    """统一依赖管理器"""
    
    def __init__(self):
        self.available: Dict[str, bool] = {}
        self.modules: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.dependencies = self._define_dependencies()
        self._check_all_dependencies()
    
    def _define_dependencies(self) -> Dict[str, DependencyInfo]:
        """定义项目依赖"""
        return {
            'numpy': DependencyInfo(
                name='numpy',
                import_name='numpy',
                description='数值计算库',
                required=True
            ),
            'scikit-optimize': DependencyInfo(
                name='scikit-optimize',
                import_name='skopt',
                description='贝叶斯优化库',
                required=False,
                alternatives=['random', 'genetic']
            ),
            'matplotlib': DependencyInfo(
                name='matplotlib',
                import_name='matplotlib.pyplot',
                description='绘图库',
                required=False
            ),
            'pandas': DependencyInfo(
                name='pandas',
                import_name='pandas',
                description='数据分析库',
                required=False
            ),
            'scipy': DependencyInfo(
                name='scipy',
                import_name='scipy',
                description='科学计算库',
                required=False
            ),
            'seaborn': DependencyInfo(
                name='seaborn',
                import_name='seaborn',
                description='统计绘图库',
                required=False
            ),
            'psutil': DependencyInfo(
                name='psutil',
                import_name='psutil',
                description='系统监控库',
                required=False
            )
        }
    
    def _check_all_dependencies(self) -> None:
        """检查所有依赖"""
        for name, dep_info in self.dependencies.items():
            try:
                module = importlib.import_module(dep_info.import_name)
                self.available[name] = True
                self.modules[name] = module
                
                # 版本检查
                if dep_info.version_check and hasattr(module, '__version__'):
                    logger.debug(f"{name} version: {module.__version__}")
                
                logger.debug(f"✓ {name} ({dep_info.description}) 可用")
                
            except ImportError as e:
                self.available[name] = False
                self.errors[name] = str(e)
                
                if dep_info.required:
                    logger.error(f"❌ 必需依赖 {name} 不可用: {e}")
                else:
                    logger.info(f"○ 可选依赖 {name} 不可用: {e}")
                    if dep_info.alternatives:
                        logger.info(f"   可用替代方案: {', '.join(dep_info.alternatives)}")
    
    def require(self, dependency: str) -> Any:
        """
        获取必需依赖
        
        Args:
            dependency: 依赖名称
            
        Returns:
            导入的模块
            
        Raises:
            ImportError: 如果必需依赖不可用
        """
        if not self.is_available(dependency):
            dep_info = self.dependencies.get(dependency)
            if dep_info and dep_info.required:
                raise ImportError(f"必需依赖 {dependency} 不可用: {self.errors.get(dependency, '未知错误')}")
            else:
                raise ImportError(f"依赖 {dependency} 不可用: {self.errors.get(dependency, '未知错误')}")
        
        return self.modules[dependency]
    
    def get_optional(self, dependency: str) -> Optional[Any]:
        """
        获取可选依赖
        
        Args:
            dependency: 依赖名称
            
        Returns:
            导入的模块或None
        """
        if self.is_available(dependency):
            return self.modules[dependency]
        return None
    
    def is_available(self, dependency: str) -> bool:
        """检查依赖是否可用"""
        return self.available.get(dependency, False)
    
    def get_available_optimizers(self) -> List[str]:
        """获取可用的优化器列表"""
        optimizers = ['random', 'genetic']  # 基础优化器
        
        if self.is_available('scikit-optimize'):
            optimizers.extend(['bayesian', 'forest'])
        
        return optimizers
    
    def get_dependency_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有依赖的状态信息"""
        status = {}
        for name, dep_info in self.dependencies.items():
            status[name] = {
                'available': self.is_available(name),
                'required': dep_info.required,
                'description': dep_info.description,
                'error': self.errors.get(name),
                'alternatives': dep_info.alternatives
            }
        return status
    
    def print_dependency_report(self) -> None:
        """打印依赖报告"""
        print("\n=== 依赖状态报告 ===")
        
        # 必需依赖
        print("\n必需依赖:")
        for name, dep_info in self.dependencies.items():
            if dep_info.required:
                status = "✓" if self.is_available(name) else "❌"
                print(f"  {status} {name}: {dep_info.description}")
                if not self.is_available(name):
                    print(f"    错误: {self.errors.get(name, '未知错误')}")
        
        # 可选依赖
        print("\n可选依赖:")
        for name, dep_info in self.dependencies.items():
            if not dep_info.required:
                status = "✓" if self.is_available(name) else "○"
                print(f"  {status} {name}: {dep_info.description}")
                if not self.is_available(name) and dep_info.alternatives:
                    print(f"    替代方案: {', '.join(dep_info.alternatives)}")
        
        # 可用优化器
        print(f"\n可用优化器: {', '.join(self.get_available_optimizers())}")

# 全局依赖管理器实例
dependency_manager = DependencyManager()

# 便捷函数
def require(dependency: str) -> Any:
    """获取必需依赖的便捷函数"""
    return dependency_manager.require(dependency)

def get_optional(dependency: str) -> Optional[Any]:
    """获取可选依赖的便捷函数"""
    return dependency_manager.get_optional(dependency)

def is_available(dependency: str) -> bool:
    """检查依赖是否可用的便捷函数"""
    return dependency_manager.is_available(dependency)

def get_available_optimizers() -> List[str]:
    """获取可用优化器的便捷函数"""
    return dependency_manager.get_available_optimizers()