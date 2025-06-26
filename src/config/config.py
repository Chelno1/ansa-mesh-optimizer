#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块 - 改进版本

作者: Chel
创建日期: 2025-06-09
版本: 1.2.0
更新日期: 2025-06-20
修复: 参数命名一致性，增强验证
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

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
        'bayesian', 'random', 'forest', 'genetic', 'parallel'
    ])
    
    # 并行配置
    n_jobs: int = 1
    use_cache: bool = True
    cache_file: str = 'optimization_cache.pkl'
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.01
    adaptive_early_stopping: bool = False
    
    # 敏感性分析配置
    sensitivity_analysis: bool = True
    sensitivity_trials: int = 5
    noise_level: float = 0.1
    
    def validate(self) -> Tuple[bool, str]:
        """验证配置有效性"""
        errors = []
        
        if self.n_calls <= 0:
            errors.append("n_calls must be positive")
        if self.n_initial_points <= 0:
            errors.append("n_initial_points must be positive")
        if self.optimizer not in self.available_optimizers:
            errors.append(f"optimizer must be one of {self.available_optimizers}")
        if self.patience <= 0:
            errors.append("patience must be positive")
        if self.min_delta < 0:
            errors.append("min_delta must be non-negative")
        if self.noise_level <= 0 or self.noise_level > 1:
            errors.append("noise_level must be between 0 and 1")
        
        is_valid = len(errors) == 0
        message = "Valid" if is_valid else "; ".join(errors)
        return is_valid, message

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
    batch_script: str = 'batch_mesh_improved.py'
    
    # 质量检查参数
    min_element_length: float = 2.0
    max_element_length: float = 8.0
    
    # 超时设置
    execution_timeout: int = 300  # 5分钟
    
    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> Tuple[bool, str]:
        """验证Ansa配置"""
        errors = []
        
        if self.min_element_length <= 0:
            errors.append("min_element_length must be positive")
        if self.max_element_length <= self.min_element_length:
            errors.append("max_element_length must be greater than min_element_length")
        if self.execution_timeout <= 0:
            errors.append("execution_timeout must be positive")
        if not self.script_dir.exists():
            errors.append(f"script_dir does not exist: {self.script_dir}")
        
        is_valid = len(errors) == 0
        message = "Valid" if is_valid else "; ".join(errors)
        return is_valid, message

@dataclass
class ParameterSpace:
    """参数空间定义 - 修复版本"""
    # 修复：将perimeter_length改为mesh_density，保持一致性
    element_size: Tuple[float, float] = (0.5, 2.0)    
    perimeter_length: Tuple[float, float] = (0.5, 8.0)  
    distortion_distance: Tuple[int, int] = (10, 30)  # 单位为百分比（%）
    general_min_target_len: Tuple[float, float] = (1.0, 2.0)  
    general_max_target_len: Tuple[float, float] = (8.0, 10.0)
    # 保留旧参数保证兼容性
    mesh_density: Tuple[float, float] = (0.5, 12.0)  
    mesh_quality_threshold: Tuple[float, float] = (0.2, 1.0)
    smoothing_iterations: Tuple[int, int] = (20, 80)
    mesh_growth_rate: Tuple[float, float] = (0.5, 1.5)
    mesh_topology: Tuple[int, int] = (1, 3)
    
    def to_skopt_space(self) -> List:
        """转换为scikit-optimize参数空间"""
        try:
            from skopt.space import Real, Integer
            return [
                Real(*self.element_size, name='element_size'),
                Real(*self.perimeter_length, name='perimeter_length'),  
                Integer(*self.distortion_distance, name='distortion_distance'),
                Real(*self.general_min_target_len, name='general_min_target_len'),
                Real(*self.general_max_target_len, name='general_max_target_len'),
                # 保留旧参数保证兼容性
                Real(*self.mesh_density, name='mesh_density'),
                Real(*self.mesh_quality_threshold, name='mesh_quality_threshold'),
                Integer(*self.smoothing_iterations, name='smoothing_iterations'),
                Real(*self.mesh_growth_rate, name='mesh_growth_rate'),
                Integer(*self.mesh_topology, name='mesh_topology')
            ]
        except ImportError:
            logger.error("scikit-optimize not available")
            raise RuntimeError("scikit-optimize is required for parameter space conversion")
    
    def get_bounds(self) -> List[Tuple]:
        """获取参数边界"""
        return [
            self.element_size,
            self.perimeter_length,   
            self.distortion_distance,
            self.general_min_target_len,
            self.general_max_target_len,
            # 保留旧参数保证兼容性
            self.mesh_density,
            self.mesh_quality_threshold,
            self.smoothing_iterations,
            self.mesh_growth_rate,
            self.mesh_topology
        ]
    
    def get_param_types(self) -> List[type]:
        """获取参数类型"""
        return [float, float, int, float, float, float, float, int, float, int]
    
    def get_param_names(self) -> List[str]:
        """获取参数名称"""
        return [
            'element_size', 'perimeter_length', 'distortion_distance',
            'general_min_target_len', 'general_max_target_len',
            'mesh_density', 'mesh_quality_threshold', 'smoothing_iterations',
            'mesh_growth_rate', 'mesh_topology'
        ]
    
    def validate_bounds(self) -> Tuple[bool, str]:
        """验证参数边界"""
        errors = []
        
        bounds = self.get_bounds()
        names = self.get_param_names()
        
        for i, (name, (low, high)) in enumerate(zip(names, bounds)):
            if low >= high:
                errors.append(f"Parameter {name}: lower bound {low} >= upper bound {high}")
            if low < 0 and name in ['element_size', 'perimeter_length', 'mesh_quality_threshold']:
                errors.append(f"Parameter {name}: lower bound {low} cannot be negative")
        
        is_valid = len(errors) == 0
        message = "Valid bounds" if is_valid else "; ".join(errors)
        return is_valid, message

class ConfigManager:
    """配置管理器 - 增强版本"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.optimization_config = OptimizationConfig()
        self.ansa_config = AnsaConfig()
        self.parameter_space = ParameterSpace()
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        
        # 验证配置
        self.validate_all_configs()
    
    def validate_all_configs(self) -> None:
        """验证所有配置"""
        # 验证优化配置
        is_valid, message = self.optimization_config.validate()
        if not is_valid:
            logger.warning(f"Optimization config validation issues: {message}")
        
        # 验证Ansa配置
        is_valid, message = self.ansa_config.validate()
        if not is_valid:
            logger.warning(f"Ansa config validation issues: {message}")
        
        # 验证参数空间
        is_valid, message = self.parameter_space.validate_bounds()
        if not is_valid:
            logger.error(f"Parameter space validation failed: {message}")
            raise ValueError(f"Invalid parameter space: {message}")
    
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
            
            # 重新验证配置
            self.validate_all_configs()
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def save_config(self, config_file: str) -> None:
        """保存配置到文件"""
        try:
            config_data = {
                'optimization': self._dataclass_to_dict(self.optimization_config),
                'ansa': self._dataclass_to_dict(self.ansa_config),
                'parameter_space': self._dataclass_to_dict(self.parameter_space),
                'metadata': {
                    'version': '1.2.0',
                    'created_by': 'ConfigManager',
                    'description': 'Ansa mesh optimizer configuration file'
                }
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
                current_value = getattr(obj, key)
                
                # 处理Path类型
                if isinstance(current_value, Path):
                    setattr(obj, key, Path(value))
                # 处理tuple类型
                elif isinstance(current_value, tuple):
                    setattr(obj, key, tuple(value))
                else:
                    setattr(obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
    
    def _dataclass_to_dict(self, obj: Any) -> Dict:
        """将数据类转换为字典"""
        result = {}
        for key, value in obj.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            elif isinstance(value, tuple):
                result[key] = list(value)
            else:
                result[key] = value
        return result
    
    def get_parameter_mapping(self) -> Dict[str, str]:
        """获取参数映射关系 - 修复版本"""
        return {
            'element_size': 'target_element_length',
            'perimeter_length': 'perimeter_length',
            'distortion_distance': 'distortion-distance',
            'general_min_target_len': 'general_min_target_len',
            'general_max_target_len': 'general_max_target_len',
            'mesh_density': 'cfd_mesh_density',
            'mesh_growth_rate': 'cfd_interior_growth_rate',
            'mesh_topology': 'mesh_type',
            'cfd_quality_threshold': 'cfd_distortion_angle',
            'cfd_min_length': 'cfd_min_length',
            'cfd_max_length': 'cfd_max_length',
            'defeaturing_length': 'defeaturing_length',
            'stl_chordal_deviation': 'stl_chordal_deviation'
        }
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'optimization': {
                'optimizer': self.optimization_config.optimizer,
                'n_calls': self.optimization_config.n_calls,
                'early_stopping': self.optimization_config.early_stopping,
                'use_cache': self.optimization_config.use_cache
            },
            'parameter_space': {
                'param_count': len(self.parameter_space.get_param_names()),
                'param_names': self.parameter_space.get_param_names(),
                'bounds_valid': self.parameter_space.validate_bounds()[0]
            },
            'ansa': {
                'executable': self.ansa_config.ansa_executable,
                'script_dir': str(self.ansa_config.script_dir),
                'timeout': self.ansa_config.execution_timeout
            }
        }
    
    def create_example_config(self, output_file: str) -> None:
        """创建示例配置文件"""
        example_config = {
            "optimization": {
                "n_calls": 50,
                "n_initial_points": 10,
                "optimizer": "bayesian",
                "early_stopping": True,
                "patience": 8,
                "use_cache": True,
                "sensitivity_analysis": True
            },
            "ansa": {
                "ansa_executable": "ansa",
                "input_model": "example_model.ansa",
                "min_element_length": 1.5,
                "max_element_length": 10.0,
                "execution_timeout": 600
            },
            "parameter_space": {
                "element_size": [0.3, 3.0],
                "perimeter_length": [0.3, 12.0],
                "distortion_distance": [5, 50],
                "general_min_target_len": [0.5, 3.0],
                "general_max_target_len": [8.0, 10.0],
                "mesh_density": [0.5, 12.0],
                "mesh_quality_threshold": [0.1, 0.9],
                "smoothing_iterations": [10, 100],
                "mesh_growth_rate": [0.3, 2.0],
                "mesh_topology": [1, 4]
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(example_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"示例配置文件已创建: {output_file}")

# 全局配置实例
config_manager = ConfigManager()

# 向后兼容性检查
def check_parameter_consistency():
    """检查参数命名一致性"""
    param_names = config_manager.parameter_space.get_param_names()
    mapping = config_manager.get_parameter_mapping()
    
    inconsistencies = []
    for param_name in param_names:
        if param_name not in mapping:
            inconsistencies.append(f"Parameter {param_name} not in mapping")
    
    if inconsistencies:
        logger.warning(f"Parameter mapping inconsistencies: {inconsistencies}")
    else:
        logger.debug("Parameter mapping is consistent")

# 执行一致性检查
check_parameter_consistency()