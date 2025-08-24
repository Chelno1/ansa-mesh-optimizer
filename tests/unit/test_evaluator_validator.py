#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试evaluator.validator模块

作者: Chel
创建日期: 2025-08-24
版本: 1.0.0
"""

import pytest
from unittest.mock import Mock, MagicMock

from ansa_mesh_optimizer.evaluators.validator import ParameterValidator


class TestParameterValidator:
    """测试ParameterValidator类"""

    def setup_method(self):
        """设置测试环境"""
        # 创建mock配置管理器
        self.mock_config_manager = Mock()
        self.mock_parameter_space = Mock()
        self.mock_validator = Mock()
        
        self.mock_config_manager.parameter_space = self.mock_parameter_space
        
        # Mock get_parameter_validator函数
        with pytest.MonkeyPatch().context() as m:
            m.setattr("src.evaluators.validator.get_parameter_validator", 
                     lambda space: self.mock_validator)
            self.validator = ParameterValidator(self.mock_config_manager)

    def test_init_success(self):
        """测试正常初始化"""
        assert self.validator.config_manager == self.mock_config_manager
        assert self.validator.validator == self.mock_validator

    def test_init_without_config_manager(self):
        """测试缺少配置管理器时的初始化"""
        with pytest.raises(ValueError, match="ParameterValidator requires a config_manager instance"):
            ParameterValidator(None)

    def test_validate_params_success(self):
        """测试参数验证成功"""
        # 设置mock返回值
        self.mock_validator.validate_comprehensive.return_value = (True, "", {})
        
        params = {"distortion_distance": 20.0}
        result = self.validator.validate_params(params)
        
        assert result is True
        self.mock_validator.validate_comprehensive.assert_called_once_with(params)

    def test_validate_params_failure(self):
        """测试参数验证失败"""
        # 设置mock返回值
        self.mock_validator.validate_comprehensive.return_value = (False, "Invalid param", {})
        
        params = {"invalid_param": -1.0}
        
        with pytest.raises(ValueError, match="Parameter validation failed: Invalid param"):
            self.validator.validate_params(params)

    def test_validate_params_exception(self):
        """测试参数验证过程中抛出异常"""
        # 设置mock抛出异常
        self.mock_validator.validate_comprehensive.side_effect = RuntimeError("Validation error")
        
        params = {"distortion_distance": 20.0}
        
        with pytest.raises(ValueError, match="Parameter validation error: Validation error"):
            self.validator.validate_params(params)

    def test_validate_and_normalize_params_success(self):
        """测试参数验证和标准化成功"""
        cleaned_params = {"distortion_distance": 20.0}
        
        # Mock normalize_params函数
        with pytest.MonkeyPatch().context() as m:
            m.setattr("src.evaluators.validator.normalize_params", 
                     lambda params: params)
            self.mock_validator.validate_comprehensive.return_value = (True, "", cleaned_params)
            
            params = {"distortion_distance": 20.0}
            is_valid, error_msg, result_params = self.validator.validate_and_normalize_params(params)
            
            assert is_valid is True
            assert error_msg == ""
            assert result_params == cleaned_params

    def test_validate_and_normalize_params_failure(self):
        """测试参数验证和标准化失败"""
        # Mock normalize_params函数
        with pytest.MonkeyPatch().context() as m:
            m.setattr("src.evaluators.validator.normalize_params", 
                     lambda params: params)
            self.mock_validator.validate_comprehensive.return_value = (False, "Invalid param", {})
            
            params = {"invalid_param": -1.0}
            is_valid, error_msg, result_params = self.validator.validate_and_normalize_params(params)
            
            assert is_valid is False
            assert "Invalid param" in error_msg
            assert result_params == {}

    def test_validate_and_normalize_params_exception(self):
        """测试参数验证和标准化过程中抛出异常"""
        # Mock normalize_params函数抛出异常
        with pytest.MonkeyPatch().context() as m:
            m.setattr("src.evaluators.validator.normalize_params", 
                     Mock(side_effect=ValueError("Normalization error")))
            
            params = {"distortion_distance": 20.0}
            is_valid, error_msg, result_params = self.validator.validate_and_normalize_params(params)
            
            assert is_valid is False
            assert "参数处理失败" in error_msg
            assert result_params == {}

    def test_validate_params_for_evaluation_success(self):
        """测试为评估验证参数成功"""
        cleaned_params = {"distortion_distance": 20.0}
        
        # Mock validate_and_normalize_params方法
        self.validator.validate_and_normalize_params = Mock(
            return_value=(True, "", cleaned_params)
        )
        
        params = {"distortion_distance": 20.0}
        is_valid, result_params = self.validator.validate_params_for_evaluation(params)
        
        assert is_valid is True
        assert result_params == cleaned_params

    def test_validate_params_for_evaluation_failure(self):
        """测试为评估验证参数失败"""
        # Mock validate_and_normalize_params方法
        self.validator.validate_and_normalize_params = Mock(
            return_value=(False, "Invalid param", {})
        )
        
        params = {"invalid_param": -1.0}
        is_valid, result_params = self.validator.validate_params_for_evaluation(params)
        
        assert is_valid is False
        assert result_params == {}