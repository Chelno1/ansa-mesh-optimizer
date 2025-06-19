#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块

作者: Chel
创建日期: 2025-06-09
版本: 1.1.0
更新日期: 2025-06-19
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from skopt.space import Real, Integer

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """优化配置类"""
    # 基础参数
    n_calls: int = 20
    n_initial_points: int = 5
    random_state: int = 42
    verbose: bool = True
    
    # 优化器配置
    optimizer: str = 'bayesian'
    available_optimizers: List[str] = field(default_factory=lambda: [
        'bayesian', 'random', 'forest', 'genetic'
    ])
    
    # 并行配置
    n_jobs: int = 1
    use_cache: bool = True
    cache_file: str = 'optimization_cache.pkl'
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.01
    
    # 敏感性分析配置
    sensitivity_analysis: bool = True
    sensitivity_trials: int = 5
    noise_level: float = 0.1

@dataclass
class AnsaConfig:
    """Ansa软件配置类"""
    # 路径配置
    ansa_executable: str = 'ansa'
    script_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    input_model: str = 'input_model.ansa'
    output_dir: Path = field(default_factory=lambda: Path('output'))
    
    # 参数文件配置
    mpar_file_pattern: str = '*.ansa_mpar'
    qual_file_pattern: str = '*.ansa_qual'
    
    # 批处理脚本
    batch_script: str = 'batch_mesh.py'
    
    # 质量检查参数
    min_element_length: float = 2.0
    max_element_length: float = 8.0
    
    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ParameterSpace:
    """参数空间定义"""
    element_size: Tuple[float, float] = (0.5, 2.0)
    mesh_density: Tuple[int, int] = (1, 5)
    mesh_quality_threshold: Tuple[float, float] = (0.2, 1.0)
    smoothing_iterations: Tuple[int, int] = (20, 80)
    mesh_growth_rate: Tuple[float, float] = (0.5, 1.5)
    mesh_topology: Tuple[int, int] = (1, 3)
    
    def to_skopt_space(self) -> List:
        """转换为scikit-optimize参数空间"""
        return [
            Real(*self.element_size, name='element_size'),
            Integer(*self.mesh_density, name='mesh_density'),
            Real(*self.mesh_quality_threshold, name='mesh_quality_threshold'),
            Integer(*self.smoothing_iterations, name='smoothing_iterations'),
            Real(*self.mesh_growth_rate, name='mesh_growth_rate'),
            Integer(*self.mesh_topology, name='mesh_topology')
        ]
    
    def get_bounds(self) -> List[Tuple]:
        """获取参数边界"""
        return [
            self.element_size,
            self.mesh_density,
            self.mesh_quality_threshold,
            self.smoothing_iterations,
            self.mesh_growth_rate,
            self.mesh_topology
        ]
    
    def get_param_types(self) -> List[type]:
        """获取参数类型"""
        return [float, int, float, int, float, int]
    
    def get_param_names(self) -> List[str]:
        """获取参数名称"""
        return [
            'element_size', 'mesh_density', 'mesh_quality_threshold',
            'smoothing_iterations', 'mesh_growth_rate', 'mesh_topology'
        ]

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.optimization_config = OptimizationConfig()
        self.ansa_config = AnsaConfig()
        self.parameter_space = ParameterSpace()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            if 'optimization' in config_data:
                self._update_dataclass(self.optimization_config, config_data['optimization'])
            
            if 'ansa' in config_data:
                self._update_dataclass(self.ansa_config, config_data['ansa'])
            
            if 'parameter_space' in config_data:
                self._update_dataclass(self.parameter_space, config_data['parameter_space'])
            
            logger.info(f"配置已从 {config_file} 加载")
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def save_config(self, config_file: str) -> None:
        """保存配置到文件"""
        try:
            config_data = {
                'optimization': self._dataclass_to_dict(self.optimization_config),
                'ansa': self._dataclass_to_dict(self.ansa_config),
                'parameter_space': self._dataclass_to_dict(self.parameter_space)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise
    
    def _update_dataclass(self, obj: Any, data: Dict) -> None:
        """更新数据类实例"""
        for key, value in data.items():
            if hasattr(obj, key):
                # 处理Path类型
                if isinstance(getattr(obj, key), Path):
                    setattr(obj, key, Path(value))
                else:
                    setattr(obj, key, value)
    
    def _dataclass_to_dict(self, obj: Any) -> Dict:
        """将数据类转换为字典"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def get_parameter_mapping(self) -> Dict[str, str]:
        """获取参数映射关系"""
        return {
            'element_size': 'target_element_length',
            'mesh_density': 'perimeter_length',
            'mesh_quality_threshold': 'distortion-angle',
            'smoothing_iterations': 'general_min_target_len',
            'mesh_growth_rate': 'cfd_interior_growth_rate',
            'mesh_topology': 'mesh_type',
            'cfd_quality_threshold': 'cfd_distortion_angle',
            'cfd_min_length': 'cfd_min_length',
            'cfd_max_length': 'cfd_max_length',
            'defeaturing_length': 'defeaturing_length',
            'stl_chordal_deviation': 'stl_chordal_deviation'
        }

# 全局配置实例
config_manager = ConfigManager()