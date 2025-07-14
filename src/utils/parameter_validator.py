#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一参数验证模块 - 消除重复代码

作者: Chel
创建日期: 2025-07-14
版本: 1.0.0
功能: 提供统一的参数验证和处理功能
"""

import logging
from typing import Dict, List, Tuple, Any, Union, Optional

logger = logging.getLogger(__name__)


class ParameterValidator:
    """统一参数验证器 - 消除重复代码"""
    
    def __init__(self, param_space):
        """
        初始化参数验证器
        
        Args:
            param_space: 参数空间对象
        """
        self.param_space = param_space
        self.bounds = param_space.get_bounds()
        self.param_names = param_space.get_parameter_names()
        self.param_types = param_space.get_parameter_types()
    
    def validate_comprehensive(self, params: Dict[str, Any], 
                             allow_partial: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """
        全面的参数验证
        
        Args:
            params: 输入参数字典
            allow_partial: 是否允许部分参数（缺失参数用默认值填充）
        
        Returns:
            (is_valid, error_message, cleaned_params)
        """
        errors = []
        cleaned_params = {}
        
        # 获取默认参数值
        default_values = self._get_default_parameter_values()
        
        # 检查和处理参数
        for name in self.param_names:
            if name not in params:
                if allow_partial and name in default_values:
                    # 使用默认值填充缺失参数
                    cleaned_params[name] = default_values[name]
                    logger.debug(f"使用默认值填充参数 {name}: {default_values[name]}")
                else:
                    errors.append(f"缺少必需参数: {name}")
                continue
            
            value = params[name]
            
            # 类型转换和验证
            try:
                cleaned_value = self._clean_and_validate_param(name, value)
                cleaned_params[name] = cleaned_value
            except ValueError as e:
                errors.append(f"参数 {name} 验证失败: {e}")
        
        # 检查额外参数
        extra_params = set(params.keys()) - set(self.param_names)
        if extra_params:
            logger.warning(f"忽略额外参数: {extra_params}")
        
        # 返回结果
        is_valid = len(errors) == 0
        error_message = "; ".join(errors) if errors else "验证通过"
        
        return is_valid, error_message, cleaned_params
    
    def _clean_and_validate_param(self, name: str, value: Any) -> Union[int, float]:
        """清理和验证单个参数"""
        param_index = self.param_names.index(name)
        expected_type = self.param_types[param_index]
        low, high = self.bounds[param_index]
        
        # 转换numpy类型
        if hasattr(value, 'item'):
            value = value.item()
        
        # 类型转换
        try:
            if hasattr(expected_type, 'value'):
                # 处理枚举类型
                if expected_type.value == 'integer':
                    cleaned_value: Union[int, float] = int(round(float(value)))
                elif expected_type.value == 'float':
                    cleaned_value = float(value)
                else:
                    cleaned_value = float(value)
            elif expected_type == int:
                cleaned_value = int(round(float(value)))
            else:
                cleaned_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"无法转换为{expected_type}: {value}")
        
        # 边界检查
        if not (low <= cleaned_value <= high):
            raise ValueError(f"值 {cleaned_value} 超出范围 [{low}, {high}]")
        
        return cleaned_value
    
    def _get_default_parameter_values(self) -> Dict[str, float]:
        """获取默认参数值"""
        try:
            # 从参数空间获取默认值
            if hasattr(self.param_space, 'get_default_values'):
                return self.param_space.get_default_values()
        except (AttributeError, Exception):
            pass
        
        # 使用硬编码默认值作为备选
        return {
            'distortion_distance': 20.0,
            'rule_fillet_width_1': 3.0,
            'rule_fillet_width_2': 10.0,
            'rule_fillet_width_3': 20.0,
            'rule_fillet_width_4': 30.0,
            'recognize_chamfers_min_angle': 20.0,
            'recognize_chamfers_max_angle': 70.0,
            'recognize_chamfers_max_width': 20.0,
            'rule_chamfer_width_1': 10.0,
            'distortion_angle': 0.0,
            'perimeter_distance': 0.667
        }
    
    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """
        标准化参数字典，将numpy类型转换为Python原生类型
        
        Args:
            params: 参数字典
            
        Returns:
            标准化后的参数字典
        """
        normalized = {}
        
        for key, value in params.items():
            try:
                if hasattr(value, 'item'):  # numpy标量类型
                    normalized[key] = value.item()
                elif hasattr(value, 'dtype'):  # numpy数组等
                    if hasattr(value, 'size') and value.size == 1:
                        normalized[key] = value.item()
                    else:
                        normalized[key] = float(value.tolist()[0]) if hasattr(value, 'tolist') else float(value)
                elif isinstance(value, (list, tuple)) and len(value) == 1:
                    # 处理单元素序列
                    normalized[key] = self.normalize_params({'temp': value[0]})['temp']
                else:
                    normalized[key] = value
            except Exception as e:
                logger.warning(f"Failed to normalize parameter {key}={value}: {e}")
                normalized[key] = value
        
        return normalized
    
    def validate_param_types(self, params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
        """
        验证并转换参数类型
        
        Args:
            params: 参数字典
            
        Returns:
            验证后的参数字典
        """
        validated_params: Dict[str, Union[int, float]] = {}
        
        try:
            for i, name in enumerate(self.param_names):
                if name in params:
                    value = params[name]
                    expected_type = self.param_types[i]
                    low, high = self.bounds[i]
                    
                    # 转换numpy类型
                    if hasattr(value, 'item'):
                        value = value.item()
                    
                    # 类型转换
                    try:
                        if hasattr(expected_type, 'value'):
                            # 处理枚举类型
                            if expected_type.value == 'integer':
                                converted_value: Union[int, float] = int(round(float(value)))
                            elif expected_type.value == 'float':
                                converted_value = float(value)
                            else:
                                converted_value = float(value)
                        elif expected_type == int:
                            converted_value = int(round(float(value)))
                        elif expected_type == float:
                            converted_value = float(value)
                        else:
                            converted_value = value
                        
                        # 边界检查
                        if low <= converted_value <= high:
                            validated_params[name] = converted_value
                        else:
                            logger.warning(f"Parameter {name}={converted_value} out of bounds [{low}, {high}]")
                            # 截断到边界内
                            validated_params[name] = max(low, min(high, converted_value))
                            
                    except (ValueError, TypeError) as e:
                        logger.error(f"Type conversion failed for {name}={value}: {e}")
                        # 使用默认值（边界中点）
                        default_value = (low + high) / 2
                        if hasattr(expected_type, 'value') and expected_type.value == 'integer':
                            default_value = int(round(default_value))
                        elif expected_type == int:
                            default_value = int(round(default_value))
                        validated_params[name] = default_value
                        
                else:
                    logger.warning(f"参数 {name} 缺失")
                    # 使用默认值
                    low, high = self.bounds[i]
                    default_value = (low + high) / 2
                    if hasattr(self.param_types[i], 'value') and self.param_types[i].value == 'integer':
                        default_value = int(round(default_value))
                    elif self.param_types[i] == int:
                        default_value = int(round(default_value))
                    validated_params[name] = default_value
            
        except Exception as e:
            logger.error(f"Parameter validation failed: {e}")
            raise ValueError(f"Parameter validation error: {e}")
        
        return validated_params


# 全局验证器实例缓存
_validator_cache = {}


def get_parameter_validator(param_space) -> ParameterValidator:
    """
    获取参数验证器实例（带缓存）
    
    Args:
        param_space: 参数空间对象
        
    Returns:
        参数验证器实例
    """
    # 使用参数空间的哈希作为缓存键
    cache_key = id(param_space)
    
    if cache_key not in _validator_cache:
        _validator_cache[cache_key] = ParameterValidator(param_space)
    
    return _validator_cache[cache_key]


def clear_validator_cache():
    """清除验证器缓存"""
    global _validator_cache
    _validator_cache.clear()


# 便捷函数
def normalize_params(params: Dict[str, Any]) -> Dict[str, Union[int, float]]:
    """
    标准化参数字典，将numpy类型转换为Python原生类型
    
    Args:
        params: 参数字典
        
    Returns:
        标准化后的参数字典
    """
    normalized = {}
    
    for key, value in params.items():
        try:
            if hasattr(value, 'item'):  # numpy标量类型
                normalized[key] = value.item()
            elif hasattr(value, 'dtype'):  # numpy数组等
                if hasattr(value, 'size') and value.size == 1:
                    normalized[key] = value.item()
                else:
                    normalized[key] = float(value.tolist()[0]) if hasattr(value, 'tolist') else float(value)
            elif isinstance(value, (list, tuple)) and len(value) == 1:
                # 处理单元素序列
                normalized[key] = normalize_params({'temp': value[0]})['temp']
            else:
                normalized[key] = value
        except Exception as e:
            logger.warning(f"Failed to normalize parameter {key}={value}: {e}")
            normalized[key] = value
    
    return normalized


def validate_param_types(params: Dict[str, Any], param_space) -> Dict[str, Union[int, float]]:
    """
    验证并转换参数类型
    
    Args:
        params: 参数字典
        param_space: 参数空间对象
        
    Returns:
        验证后的参数字典
    """
    validator = get_parameter_validator(param_space)
    return validator.validate_param_types(params)