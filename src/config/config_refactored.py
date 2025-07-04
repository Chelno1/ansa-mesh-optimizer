#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重构后的配置管理模块

作者: Chel
创建日期: 2025-07-04
版本: 1.3.0
功能: 消除参数重复，统一配置管理，集成异常处理和依赖管理
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json
import logging
from enum import Enum

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.utils.exceptions import ConfigurationError, ValidationError, handle_exceptions
    from src.utils.dependency_manager import is_available
except ImportError:
    # 如果从src目录运行，使用相对导入
    from utils.exceptions import ConfigurationError, ValidationError, handle_exceptions
    from utils.dependency_manager import is_available

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """优化器类型枚举"""
    BAYESIAN = "bayesian"
    RANDOM = "random"
    FOREST = "forest"
    GENETIC = "genetic"
    PARALLEL = "parallel"


class ParameterType(Enum):
    """参数类型枚举"""
    FLOAT = "float"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass
class ParameterDefinition:
    """参数定义类"""
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
                return isinstance(value, (int, float)) and isinstance(low, (int, float)) and isinstance(high, (int, float)) and low <= value <= high
        elif self.param_type == ParameterType.INTEGER:
            if isinstance(self.bounds, tuple) and len(self.bounds) == 2:
                low, high = self.bounds
                return isinstance(value, int) and isinstance(low, int) and isinstance(high, int) and low <= value <= high
        elif self.param_type == ParameterType.CATEGORICAL:
            if isinstance(self.bounds, list):
                return value in self.bounds
        return False


@dataclass
class OptimizationConfig:
    """优化配置类 - 重构版"""
    # 基础优化参数
    n_calls: int = 20
    n_initial_points: int = 5
    random_state: int = 42
    verbose: bool = True
    
    # 优化器配置
    optimizer: OptimizerType = OptimizerType.BAYESIAN
    
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
    
    # 收敛配置
    convergence_threshold: float = 1e-6
    max_stagnation_iterations: int = 10
    
    def get_available_optimizers(self) -> List[str]:
        """获取可用的优化器列表"""
        available = [OptimizerType.RANDOM.value, OptimizerType.GENETIC.value]
        
        if is_available('scikit-optimize'):
            available.extend([OptimizerType.BAYESIAN.value, OptimizerType.FOREST.value])
        
        return available
    
    @handle_exceptions()
    def validate(self) -> None:
        """验证配置有效性"""
        errors = []
        
        if self.n_calls <= 0:
            errors.append("n_calls must be positive")
        if self.n_initial_points <= 0:
            errors.append("n_initial_points must be positive")
        if self.n_initial_points >= self.n_calls:
            errors.append("n_initial_points must be less than n_calls")
        available_optimizers = self.get_available_optimizers()
        if self.optimizer.value not in available_optimizers:
            # 自动调整到可用的优化器
            if available_optimizers:
                old_optimizer = self.optimizer.value
                # 选择第一个可用的优化器
                new_optimizer_value = available_optimizers[0]
                # 找到对应的枚举值
                for opt_type in OptimizerType:
                    if opt_type.value == new_optimizer_value:
                        self.optimizer = opt_type
                        break
                logger.warning(f"优化器 {old_optimizer} 不可用，自动切换到 {new_optimizer_value}")
            else:
                errors.append(f"no optimizers available")
        if self.patience <= 0:
            errors.append("patience must be positive")
        if self.min_delta < 0:
            errors.append("min_delta must be non-negative")
        if not 0 < self.noise_level <= 1:
            errors.append("noise_level must be between 0 and 1")
        if self.convergence_threshold <= 0:
            errors.append("convergence_threshold must be positive")
        if self.max_stagnation_iterations <= 0:
            errors.append("max_stagnation_iterations must be positive")
        
        if errors:
            raise ConfigurationError(f"Optimization config validation failed: {'; '.join(errors)}")


@dataclass
class AnsaConfig:
    """Ansa软件配置类 - 重构版"""
    # 路径配置
    ansa_executable: str = 'ansa'
    script_dir: Path = field(default_factory=lambda: Path(__file__).parent)
    input_model: str = 'input_model.ansa'
    output_dir: Path = field(default_factory=lambda: Path('output'))
    
    # 文件模式配置
    mpar_file_pattern: str = '*.ansa_mpar'
    qual_file_pattern: str = '*.ansa_qual'
    batch_script: str = 'batch_mesh_improved.py'
    
    # 执行配置
    execution_timeout: int = 300  # 5分钟
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 质量检查参数
    min_element_length: float = 2.0
    max_element_length: float = 8.0
    quality_check_enabled: bool = True
    
    # 内存和性能配置
    max_memory_usage: float = 8.0  # GB
    temp_cleanup: bool = True
    
    def __post_init__(self):
        """确保输出目录存在"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @handle_exceptions()
    def validate(self) -> None:
        """验证Ansa配置"""
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
        if not self.script_dir.exists():
            errors.append(f"script_dir does not exist: {self.script_dir}")
        
        if errors:
            raise ConfigurationError(f"Ansa config validation failed: {'; '.join(errors)}")


@dataclass
class UnifiedParameterSpace:
    """统一参数空间定义 - 消除重复"""
    
    def __init__(self, config_specified_params: Optional[List[str]] = None):
        """
        初始化参数空间
        
        Args:
            config_specified_params: 配置文件中指定的参数列表，如果提供则只使用这些参数
        """
        self.config_specified_params = config_specified_params
        self.parameters = self._define_parameters()
        
        # 如果指定了配置文件参数，则只保留这些参数
        if config_specified_params is not None:
            filtered_params = {}
            for param_name in config_specified_params:
                if param_name in self.parameters:
                    filtered_params[param_name] = self.parameters[param_name]
                else:
                    logger.warning(f"配置文件中指定的参数 '{param_name}' 不在默认参数空间中")
            self.parameters = filtered_params
            logger.info(f"使用配置文件指定的参数: {list(self.parameters.keys())}")
    
    def _define_parameters(self) -> Dict[str, ParameterDefinition]:
        """定义统一的参数空间"""
        return {
            # 网格尺寸参数
            'element_size': ParameterDefinition(
                name='element_size',
                param_type=ParameterType.FLOAT,
                bounds=(0.5, 2.0),
                description='目标单元尺寸',
                unit='mm',
                ansa_mapping='target_element_length',
                default_value=1.0
            ),
            'perimeter_length': ParameterDefinition(
                name='perimeter_length',
                param_type=ParameterType.FLOAT,
                bounds=(0.5, 8.0),
                description='周边长度',
                unit='mm',
                ansa_mapping='perimeter_length',
                default_value=2.0
            ),
            'min_target_length': ParameterDefinition(
                name='min_target_length',
                param_type=ParameterType.FLOAT,
                bounds=(1.0, 2.0),
                description='最小目标长度',
                unit='mm',
                ansa_mapping='general_min_target_len',
                default_value=1.5
            ),
            'max_target_length': ParameterDefinition(
                name='max_target_length',
                param_type=ParameterType.FLOAT,
                bounds=(8.0, 10.0),
                description='最大目标长度',
                unit='mm',
                ansa_mapping='general_max_target_len',
                default_value=9.0
            ),
            
            # 网格质量参数
            'distortion_distance': ParameterDefinition(
                name='distortion_distance',
                param_type=ParameterType.INTEGER,
                bounds=(10, 30),
                description='扭曲距离',
                unit='%',
                ansa_mapping='distortion-distance',
                default_value=20
            ),
            'quality_threshold': ParameterDefinition(
                name='quality_threshold',
                param_type=ParameterType.FLOAT,
                bounds=(0.2, 1.0),
                description='质量阈值',
                unit='',
                ansa_mapping='cfd_distortion_angle',
                default_value=0.6
            ),
            'smoothing_iterations': ParameterDefinition(
                name='smoothing_iterations',
                param_type=ParameterType.INTEGER,
                bounds=(20, 80),
                description='平滑迭代次数',
                unit='',
                ansa_mapping='smoothing_iterations',
                default_value=50
            ),
            
            # CFD特定参数
            'mesh_density': ParameterDefinition(
                name='mesh_density',
                param_type=ParameterType.FLOAT,
                bounds=(0.5, 12.0),
                description='网格密度',
                unit='',
                ansa_mapping='cfd_mesh_density',
                default_value=5.0
            ),
            'growth_rate': ParameterDefinition(
                name='growth_rate',
                param_type=ParameterType.FLOAT,
                bounds=(0.5, 1.5),
                description='增长率',
                unit='',
                ansa_mapping='cfd_interior_growth_rate',
                default_value=1.0
            ),
            'mesh_topology': ParameterDefinition(
                name='mesh_topology',
                param_type=ParameterType.INTEGER,
                bounds=(1, 3),
                description='网格拓扑类型',
                unit='',
                ansa_mapping='mesh_type',
                default_value=2
            )
        }
    
    def get_parameter(self, name: str) -> Optional[ParameterDefinition]:
        """获取参数定义"""
        return self.parameters.get(name)
    
    def get_parameter_names(self) -> List[str]:
        """获取所有参数名称"""
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
        """转换为scikit-optimize参数空间"""
        if not is_available('scikit-optimize'):
            raise ConfigurationError("scikit-optimize not available for parameter space conversion")
        
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
        except ImportError as e:
            raise ConfigurationError(f"Failed to create skopt space: {e}")
    
    @handle_exceptions()
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
                        if low < 0 and name in ['element_size', 'perimeter_length', 'quality_threshold']:
                            errors.append(f"Parameter {name}: lower bound {low} cannot be negative")
        
        if errors:
            raise ValidationError(f"Parameter bounds validation failed: {'; '.join(errors)}")
    
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
        
        if errors:
            raise ValidationError(f"Parameter values validation failed: {'; '.join(errors)}")


class UnifiedConfigManager:
    """统一配置管理器 - 重构版"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        
        # 如果提供了配置文件，先提取参数列表
        config_specified_params = None
        if config_file and Path(config_file).exists():
            config_specified_params = self._extract_config_parameters(config_file)
        
        # 初始化配置对象
        self.optimization_config = OptimizationConfig()
        self.ansa_config = AnsaConfig()
        
        # 初始化参数空间（如果有配置文件，只使用配置文件中指定的参数）
        self.parameter_space = UnifiedParameterSpace(config_specified_params)
        
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
        
        # 验证所有配置
        self.validate_all_configs()
    
    def _extract_config_parameters(self, config_file: str) -> Optional[List[str]]:
        """
        从配置文件中提取参数列表
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            配置文件中指定的参数名称列表，如果没有parameters部分则返回None
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            if 'parameters' in config_data:
                param_names = list(config_data['parameters'].keys())
                logger.info(f"从配置文件提取到参数: {param_names}")
                return param_names
            else:
                logger.info("配置文件中未找到parameters部分，将使用所有默认参数")
                return None
                
        except Exception as e:
            logger.warning(f"提取配置文件参数失败: {e}，将使用所有默认参数")
            return None
    
    @handle_exceptions()
    def validate_all_configs(self) -> None:
        """验证所有配置"""
        try:
            self.optimization_config.validate()
            self.ansa_config.validate()
            self.parameter_space.validate_bounds()
            logger.info("所有配置验证通过")
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    @handle_exceptions()
    def load_config(self, config_file: str) -> None:
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新配置
            if 'optimization' in config_data:
                self._update_optimization_config(config_data['optimization'])
            
            if 'ansa' in config_data:
                self._update_ansa_config(config_data['ansa'])
            
            if 'parameters' in config_data:
                self._update_parameter_space(config_data['parameters'])
            
            logger.info(f"配置已从 {config_file} 加载")
            
        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {e}", file_path=config_file)
    
    def _update_optimization_config(self, data: Dict) -> None:
        """更新优化配置"""
        for key, value in data.items():
            if hasattr(self.optimization_config, key):
                if key == 'optimizer':
                    # 处理优化器枚举
                    self.optimization_config.optimizer = OptimizerType(value)
                else:
                    setattr(self.optimization_config, key, value)
    
    def _update_ansa_config(self, data: Dict) -> None:
        """更新ANSA配置"""
        for key, value in data.items():
            if hasattr(self.ansa_config, key):
                current_value = getattr(self.ansa_config, key)
                if isinstance(current_value, Path):
                    setattr(self.ansa_config, key, Path(value))
                else:
                    setattr(self.ansa_config, key, value)
    
    def _update_parameter_space(self, data: Dict) -> None:
        """更新参数空间"""
        for name, config in data.items():
            if name in self.parameter_space.parameters:
                param = self.parameter_space.parameters[name]
                if 'bounds' in config:
                    param.bounds = tuple(config['bounds'])
                if 'default_value' in config:
                    param.default_value = config['default_value']
    
    @handle_exceptions()
    def save_config(self, config_file: str) -> None:
        """保存配置到文件"""
        try:
            config_data = {
                'optimization': self._optimization_config_to_dict(),
                'ansa': self._ansa_config_to_dict(),
                'parameters': self._parameter_space_to_dict(),
                'metadata': {
                    'version': '2.0.0',
                    'created_by': 'UnifiedConfigManager',
                    'description': 'Unified Ansa mesh optimizer configuration file'
                }
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存到 {config_file}")
            
        except Exception as e:
            raise ConfigurationError(f"保存配置文件失败: {e}", file_path=config_file)
    
    def _optimization_config_to_dict(self) -> Dict:
        """优化配置转字典"""
        result = {}
        for key, value in self.optimization_config.__dict__.items():
            if isinstance(value, OptimizerType):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    def _ansa_config_to_dict(self) -> Dict:
        """ANSA配置转字典"""
        result = {}
        for key, value in self.ansa_config.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def _parameter_space_to_dict(self) -> Dict:
        """参数空间转字典"""
        result = {}
        for name, param in self.parameter_space.parameters.items():
            result[name] = {
                'param_type': param.param_type.value,
                'bounds': list(param.bounds),
                'description': param.description,
                'unit': param.unit,
                'ansa_mapping': param.ansa_mapping,
                'default_value': param.default_value
            }
        return result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'optimization': {
                'optimizer': self.optimization_config.optimizer.value,
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


# 全局配置实例
unified_config_manager = UnifiedConfigManager()