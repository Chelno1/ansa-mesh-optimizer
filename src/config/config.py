#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的配置管理模块 - 替换原有复杂配置
 
作者: Chel
创建日期: 2025-07-14
版本: 2.0.0
功能: 提供简化的配置管理，减少复杂性，保持向后兼容性
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """优化器类型枚举"""
    BAYESIAN = "bayesian"
    RANDOM = "random"
    FOREST = "forest"
    GENETIC = "genetic"
    PARALLEL = "parallel"


class ParameterType(Enum):
    """参数类型枚举 - 保持向后兼容"""
    FLOAT = "float"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass
class SimpleOptimizationConfig:
    """简化的优化配置"""
    n_calls: int = 20
    n_initial_points: int = 5
    random_state: int = 42
    verbose: bool = True
    optimizer: str = "genetic"  # 使用字符串而不是枚举
    use_cache: bool = True
    cache_file: str = 'optimization_cache.pkl'
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.01
    sensitivity_analysis: bool = True
    sensitivity_trials: int = 5
    noise_level: float = 0.1
    
    # 新增字段以保持向后兼容
    n_jobs: int = 1
    adaptive_early_stopping: bool = False
    convergence_threshold: float = 1e-6
    max_stagnation_iterations: int = 10
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """简化的验证，返回元组以保持兼容性"""
        errors = []
        
        if self.n_calls <= 0:
            errors.append("n_calls must be positive")
        if self.n_initial_points <= 0:
            errors.append("n_initial_points must be positive")
        if self.n_initial_points >= self.n_calls:
            errors.append("n_initial_points must be less than n_calls")
        if self.patience <= 0:
            errors.append("patience must be positive")
        if self.min_delta < 0:
            errors.append("min_delta must be non-negative")
        if not 0 < self.noise_level <= 1:
            errors.append("noise_level must be between 0 and 1")
        
        if errors:
            return False, "; ".join(errors)
        return True, None
    
    def get_available_optimizers(self) -> List[str]:
        """获取可用的优化器列表"""
        available = ["random", "genetic"]
        
        try:
            import skopt
            available.extend(["bayesian", "forest"])
        except ImportError:
            pass
        
        return available


@dataclass
class SimpleAnsaConfig:
    """简化的ANSA配置"""
    ansa_executable: str = 'ansa'
    script_dir: Path = field(default_factory=lambda: Path('src'))
    input_model: str = 'input_model.ansa'
    output_dir: Path = field(default_factory=lambda: Path('output'))
    mpar_file_pattern: str = '*.ansa_mpar'
    qual_file_pattern: str = '*.ansa_qual'
    batch_script: str = 'batch_mesh.py'
    execution_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 1.0
    min_element_length: float = 2.0
    max_element_length: float = 8.0
    quality_check_enabled: bool = True
    max_memory_usage: float = 8.0
    temp_cleanup: bool = True
    
    def validate(self) -> Tuple[bool, Optional[str]]:
        """简化的验证，返回元组以保持兼容性"""
        errors = []
        
        if self.min_element_length <= 0:
            errors.append("min_element_length must be positive")
        if self.max_element_length <= self.min_element_length:
            errors.append("max_element_length must be greater than min_element_length")
        if self.execution_timeout <= 0:
            errors.append("execution_timeout must be positive")
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        if self.retry_delay < 0:
            errors.append("retry_delay must be non-negative")
        if self.max_memory_usage <= 0:
            errors.append("max_memory_usage must be positive")
        
        if errors:
            return False, "; ".join(errors)
        return True, None
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ParameterDefinition:
    """参数定义类 - 保持向后兼容"""
    name: str
    param_type: ParameterType
    bounds: Union[Tuple[float, float], Tuple[int, int], List[str]]
    description: str
    unit: Optional[str] = None
    ansa_mapping: Optional[str] = None
    default_value: Optional[Union[float, int, str]] = None
    
    def validate_value(self, value: Any) -> bool:
        """验证参数值是否在有效范围内"""
        if self.param_type == ParameterType.FLOAT:
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                low, high = self.bounds
                return isinstance(value, (int, float)) and low <= value <= high
        elif self.param_type == ParameterType.INTEGER:
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                low, high = self.bounds
                return isinstance(value, int) and low <= value <= high
        elif self.param_type == ParameterType.CATEGORICAL:
            if isinstance(self.bounds, list):
                return value in self.bounds
        return False


class SimpleParameterSpace:
    """简化的参数空间"""
    
    def __init__(self, config_specified_params: Optional[List[str]] = None):
        self.config_specified_params = config_specified_params
        self.parameters = self._define_default_parameters()
        
        # 如果指定了配置参数，只保留这些参数
        if config_specified_params:
            filtered = {name: param for name, param in self.parameters.items() 
                       if name in config_specified_params}
            if filtered:
                self.parameters = filtered
                logger.info(f"使用指定参数: {list(self.parameters.keys())}")
    
    def _define_default_parameters(self) -> Dict[str, ParameterDefinition]:
        """定义默认参数"""
        return {
            'distortion_distance': ParameterDefinition(
                name='distortion_distance',
                param_type=ParameterType.FLOAT,
                bounds=(10.0, 30.0),
                description='扭曲距离',
                unit='%',
                ansa_mapping='distortion-distance',
                default_value=20.0
            ),
            'rule_fillet_width_1': ParameterDefinition(
                name='rule_fillet_width_1',
                param_type=ParameterType.FLOAT,
                bounds=(1.0, 5.0),
                description='Fillet规则1的width上限值',
                unit='mm',
                ansa_mapping='rule_fillet_width_1',
                default_value=3.0
            ),
            'rule_fillet_width_2': ParameterDefinition(
                name='rule_fillet_width_2',
                param_type=ParameterType.FLOAT,
                bounds=(5.0, 12.0),
                description='Fillet规则2的width上限值',
                unit='mm',
                ansa_mapping='rule_fillet_width_2',
                default_value=10.0
            ),
            'rule_fillet_width_3': ParameterDefinition(
                name='rule_fillet_width_3',
                param_type=ParameterType.FLOAT,
                bounds=(12.0, 25.0),
                description='Fillet规则3的width上限值',
                unit='mm',
                ansa_mapping='rule_fillet_width_3',
                default_value=20.0
            ),
            'rule_fillet_width_4': ParameterDefinition(
                name='rule_fillet_width_4',
                param_type=ParameterType.FLOAT,
                bounds=(25.0, 40.0),
                description='Fillet规则4的width上限值',
                unit='mm',
                ansa_mapping='rule_fillet_width_4',
                default_value=30.0
            ),
            'recognize_chamfers_min_angle': ParameterDefinition(
                name='recognize_chamfers_min_angle',
                param_type=ParameterType.FLOAT,
                bounds=(10.0, 30.0),
                description='Chamfer识别的最小角度',
                unit='degrees',
                ansa_mapping='recognize_chamfers_min_angle',
                default_value=20.0
            ),
            'recognize_chamfers_max_angle': ParameterDefinition(
                name='recognize_chamfers_max_angle',
                param_type=ParameterType.FLOAT,
                bounds=(60.0, 80.0),
                description='Chamfer识别的最大角度',
                unit='degrees',
                ansa_mapping='recognize_chamfers_max_angle',
                default_value=70.0
            ),
            'recognize_chamfers_max_width': ParameterDefinition(
                name='recognize_chamfers_max_width',
                param_type=ParameterType.FLOAT,
                bounds=(10.0, 30.0),
                description='Chamfer识别的最大宽度',
                unit='mm',
                ansa_mapping='recognize_chamfers_max_width',
                default_value=20.0
            ),
            'rule_chamfer_width_1': ParameterDefinition(
                name='rule_chamfer_width_1',
                param_type=ParameterType.FLOAT,
                bounds=(5.0, 20.0),
                description='Chamfer规则的width上限值',
                unit='mm',
                ansa_mapping='rule_chamfer_width_1',
                default_value=10.0
            ),
            'distortion_angle': ParameterDefinition(
                name='distortion_angle',
                param_type=ParameterType.FLOAT,
                bounds=(0.0, 45.0),
                description='扭曲角度参数',
                unit='degrees',
                ansa_mapping='distortion-angle',
                default_value=0.0
            ),
            'perimeter_distance': ParameterDefinition(
                name='perimeter_distance',
                param_type=ParameterType.FLOAT,
                bounds=(0.667, 1.0),
                description='周边距离系数',
                unit='*Lmin',
                ansa_mapping='remove_perimeters_with_distance',
                default_value=0.667
            ),
        }
    
    def get_parameter(self, name: str) -> Optional[ParameterDefinition]:
        """获取参数定义"""
        return self.parameters.get(name)
    
    def get_parameter_names(self) -> List[str]:
        """获取参数名称"""
        return list(self.parameters.keys())
    
    def get_bounds(self) -> List[Union[Tuple[float, float], Tuple[int, int], List[str]]]:
        """获取参数边界"""
        return [param.bounds for param in self.parameters.values()]
    
    def get_parameter_types(self) -> List[ParameterType]:
        """获取参数类型"""
        return [param.param_type for param in self.parameters.values()]
    
    def get_ansa_mapping(self) -> Dict[str, str]:
        """获取ANSA参数映射"""
        return {name: param.ansa_mapping for name, param in self.parameters.items() 
                if param.ansa_mapping}
    
    def get_default_values(self) -> Dict[str, Any]:
        """获取默认值"""
        return {name: param.default_value for name, param in self.parameters.items() 
                if param.default_value is not None}
    
    def to_skopt_space(self) -> List:
        """转换为scikit-optimize空间"""
        try:
            from skopt.space import Real, Integer, Categorical
            
            space = []
            for name, param in self.parameters.items():
                if param.param_type == ParameterType.FLOAT:
                    if isinstance(param.bounds, tuple) and len(param.bounds) == 2:
                        low, high = param.bounds
                        space.append(Real(low, high, name=name))
                elif param.param_type == ParameterType.INTEGER:
                    if isinstance(param.bounds, tuple) and len(param.bounds) == 2:
                        low, high = param.bounds
                        space.append(Integer(low, high, name=name))
                elif param.param_type == ParameterType.CATEGORICAL:
                    if isinstance(param.bounds, list):
                        space.append(Categorical(param.bounds, name=name))
            
            return space
        except ImportError:
            logger.warning("scikit-optimize不可用")
            return []
    
    def validate_bounds(self) -> None:
        """验证参数边界"""
        errors = []
        
        for name, param in self.parameters.items():
            if param.param_type in [ParameterType.FLOAT, ParameterType.INTEGER]:
                if isinstance(param.bounds, tuple) and len(param.bounds) == 2:
                    low, high = param.bounds
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
                        if low >= high:
                            errors.append(f"Parameter {name}: lower bound {low} >= upper bound {high}")
        
        if errors:
            raise ValueError(f"Parameter bounds validation failed: {'; '.join(errors)}")
    
    def validate_parameter_values(self, values: Dict[str, Any]) -> None:
        """验证参数值"""
        errors = []
        
        for name, value in values.items():
            if name not in self.parameters:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            param = self.parameters[name]
            if not param.validate_value(value):
                errors.append(f"Parameter {name}: value {value} not in valid range {param.bounds}")
        
        # 验证 rule_fillet_width 参数的递增关系
        self._validate_rule_fillet_order(values, errors)
        
        if errors:
            raise ValueError(f"Parameter values validation failed: {'; '.join(errors)}")
    
    def _validate_rule_fillet_order(self, values: Dict[str, Any], errors: List[str]) -> None:
        """验证 rule_fillet_width 参数的递增顺序"""
        fillet_params = {}
        
        # 收集所有 rule_fillet_width 参数
        for i in range(1, 5):
            param_name = f'rule_fillet_width_{i}'
            if param_name in values:
                fillet_params[i] = float(values[param_name])
        
        # 检查递增顺序
        if len(fillet_params) > 1:
            sorted_indices = sorted(fillet_params.keys())
            for i in range(len(sorted_indices) - 1):
                current_idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                
                current_value = fillet_params[current_idx]
                next_value = fillet_params[next_idx]
                
                if current_value >= next_value:
                    errors.append(f"rule_fillet_width_{current_idx} ({current_value}) must be less than rule_fillet_width_{next_idx} ({next_value})")


class SimpleConfigManager:
    """简化的配置管理器 - 保持向后兼容性"""
    
    def __init__(self, config_file: Optional[str] = None, require_config: bool = False):
        self.config_file = config_file
        self.require_config = require_config
        
        # 如果要求必须有配置文件但没有提供，则抛出异常
        if require_config and not config_file:
            raise ValueError("未指定配置文件。请使用 --config 参数指定配置文件路径。")
        
        if require_config and config_file and not Path(config_file).exists():
            raise ValueError(f"配置文件不存在: {config_file}")
        
        # 加载配置文件（如果提供）
        config_params = None
        if config_file and Path(config_file).exists():
            config_params = self._load_config_file(config_file)
        
        # 初始化配置对象
        self.optimization_config = SimpleOptimizationConfig()
        self.ansa_config = SimpleAnsaConfig()
        
        # 初始化参数空间
        self.parameter_space = SimpleParameterSpace(config_params)
        
        # 验证配置
        self._validate_configs()
    
    def _load_config_file(self, config_file: str) -> Optional[List[str]]:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新优化配置
            if 'optimization' in config_data:
                opt_data = config_data['optimization']
                for key, value in opt_data.items():
                    if hasattr(self.optimization_config, key):
                        setattr(self.optimization_config, key, value)
            
            # 更新ANSA配置
            if 'ansa' in config_data:
                ansa_data = config_data['ansa']
                for key, value in ansa_data.items():
                    if hasattr(self.ansa_config, key):
                        current_value = getattr(self.ansa_config, key)
                        if isinstance(current_value, Path):
                            setattr(self.ansa_config, key, Path(value))
                        else:
                            setattr(self.ansa_config, key, value)
            
            # 返回参数列表
            if 'parameters' in config_data:
                return list(config_data['parameters'].keys())
            
            logger.info(f"配置已从 {config_file} 加载")
            return None
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return None
    
    def _validate_configs(self):
        """验证配置"""
        opt_valid, opt_error = self.optimization_config.validate()
        if not opt_valid:
            logger.warning(f"优化配置验证失败: {opt_error}")
        
        ansa_valid, ansa_error = self.ansa_config.validate()
        if not ansa_valid:
            logger.warning(f"ANSA配置验证失败: {ansa_error}")
        
        try:
            self.parameter_space.validate_bounds()
        except Exception as e:
            logger.warning(f"参数空间验证失败: {e}")
        
        logger.info("配置验证完成")
    
    def validate_all_configs(self) -> None:
        """验证所有配置 - 保持向后兼容"""
        self._validate_configs()
    
    def load_config(self, config_file: str) -> None:
        """从文件加载配置 - 保持向后兼容"""
        self._load_config_file(config_file)
    
    def save_config(self, config_file: str):
        """保存配置到文件"""
        try:
            config_data = {
                'optimization': self.optimization_config.__dict__,
                'ansa': {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in self.ansa_config.__dict__.items()
                },
                'parameters': {
                    name: {
                        'param_type': param.param_type.value,
                        'bounds': list(param.bounds),
                        'description': param.description,
                        'unit': param.unit,
                        'ansa_mapping': param.ansa_mapping,
                        'default_value': param.default_value
                    }
                    for name, param in self.parameter_space.parameters.items()
                },
                'metadata': {
                    'version': '2.0.0',
                    'created_by': 'SimpleConfigManager',
                    'description': 'Simplified Ansa mesh optimizer configuration'
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {config_file}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'optimization': {
                'optimizer': self.optimization_config.optimizer,
                'n_calls': self.optimization_config.n_calls,
                'early_stopping': self.optimization_config.early_stopping,
                'available_optimizers': self.optimization_config.get_available_optimizers()
            },
            'parameter_space': {
                'param_count': len(self.parameter_space.get_parameter_names()),
                'param_names': self.parameter_space.get_parameter_names(),
                'has_ansa_mapping': bool(self.parameter_space.get_ansa_mapping())
            },
            'ansa': {
                'executable': self.ansa_config.ansa_executable,
                'script_dir': str(self.ansa_config.script_dir),
                'timeout': self.ansa_config.execution_timeout,
                'quality_check': self.ansa_config.quality_check_enabled
            }
        }


# 向后兼容性别名
UnifiedConfigManager = SimpleConfigManager
OptimizationConfig = SimpleOptimizationConfig
AnsaConfig = SimpleAnsaConfig
UnifiedParameterSpace = SimpleParameterSpace

# 全局配置实例
unified_config_manager = None  # 将在需要时初始化