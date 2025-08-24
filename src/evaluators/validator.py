#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数验证模块
负责处理网格评估器的参数验证逻辑

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import logging
from typing import Dict, Tuple

from .utils import normalize_params
from ..utils.parameter_validator import get_parameter_validator

logger = logging.getLogger(__name__)


class ParameterValidator:
    """参数验证器 - 统一的参数验证逻辑"""

    def __init__(self, config_manager):
        """
        初始化参数验证器
        
        Args:
            config_manager: 配置管理器实例
        """
        if config_manager is None:
            raise ValueError("ParameterValidator requires a config_manager instance")
        
        self.config_manager = config_manager
        self.validator = get_parameter_validator(config_manager.parameter_space)

    def validate_params(self, params: Dict[str, float]) -> bool:
        """
        验证参数有效性
        
        Args:
            params: 网格参数字典
            
        Returns:
            参数是否有效
            
        Raises:
            ValueError: 参数验证失败时抛出
        """
        try:
            is_valid, error_msg, _ = self.validator.validate_comprehensive(params)
            if not is_valid:
                raise ValueError(f"Parameter validation failed: {error_msg}")
            return is_valid
        except ValueError:
            # 重新抛出ValueError以便调用者能够捕获
            raise
        except Exception as e:
            logger.error(f"参数验证异常: {e}")
            raise ValueError(f"Parameter validation error: {e}")

    def validate_and_normalize_params(
        self, params: Dict[str, float]
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        验证并标准化参数
        
        Args:
            params: 原始参数字典
            
        Returns:
            (is_valid, error_msg, cleaned_params) 元组
        """
        try:
            # 标准化参数
            normalized_params = normalize_params(params)
            
            # 验证参数
            is_valid, error_msg, cleaned_params = self.validator.validate_comprehensive(
                normalized_params
            )
            
            if not is_valid:
                logger.error(f"参数验证失败: {error_msg}")
                return False, error_msg, {}
                
            return True, "", cleaned_params
            
        except Exception as e:
            error_msg = f"参数处理失败: {e}"
            logger.error(error_msg)
            return False, error_msg, {}

    def validate_params_for_evaluation(
        self, params: Dict[str, float]
    ) -> Tuple[bool, Dict[str, float]]:
        """
        为评估准备和验证参数
        
        Args:
            params: 输入参数字典
            
        Returns:
            (is_valid, cleaned_params) 元组
        """
        is_valid, error_msg, cleaned_params = self.validate_and_normalize_params(params)
        
        if not is_valid:
            logger.error(f"评估参数验证失败: {error_msg}")
            return False, {}
            
        return True, cleaned_params