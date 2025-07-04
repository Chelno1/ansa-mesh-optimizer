#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块单元测试

作者: Chel
创建日期: 2025-07-04
版本: 1.3.0
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.config_refactored import (
    OptimizationConfig, AnsaConfig, UnifiedParameterSpace, 
    UnifiedConfigManager, OptimizerType, ParameterType, ParameterDefinition
)
from src.utils.exceptions import ConfigurationError, ValidationError


class TestParameterDefinition(unittest.TestCase):
    """参数定义测试"""
    
    def setUp(self):
        self.float_param = ParameterDefinition(
            name='test_float',
            param_type=ParameterType.FLOAT,
            bounds=(0.5, 2.0),
            description='Test float parameter'
        )
        
        self.int_param = ParameterDefinition(
            name='test_int',
            param_type=ParameterType.INTEGER,
            bounds=(1, 10),
            description='Test integer parameter'
        )
        
        self.cat_param = ParameterDefinition(
            name='test_cat',
            param_type=ParameterType.CATEGORICAL,
            bounds=['option1', 'option2', 'option3'],
            description='Test categorical parameter'
        )
    
    def test_validate_float_value(self):
        """测试浮点参数验证"""
        self.assertTrue(self.float_param.validate_value(1.0))
        self.assertTrue(self.float_param.validate_value(0.5))
        self.assertTrue(self.float_param.validate_value(2.0))
        self.assertFalse(self.float_param.validate_value(0.4))
        self.assertFalse(self.float_param.validate_value(2.1))
        self.assertFalse(self.float_param.validate_value('invalid'))
    
    def test_validate_int_value(self):
        """测试整数参数验证"""
        self.assertTrue(self.int_param.validate_value(5))
        self.assertTrue(self.int_param.validate_value(1))
        self.assertTrue(self.int_param.validate_value(10))
        self.assertFalse(self.int_param.validate_value(0))
        self.assertFalse(self.int_param.validate_value(11))
        self.assertFalse(self.int_param.validate_value(5.5))
    
    def test_validate_categorical_value(self):
        """测试分类参数验证"""
        self.assertTrue(self.cat_param.validate_value('option1'))
        self.assertTrue(self.cat_param.validate_value('option2'))
        self.assertFalse(self.cat_param.validate_value('invalid_option'))
        self.assertFalse(self.cat_param.validate_value(1))


class TestOptimizationConfig(unittest.TestCase):
    """优化配置测试"""
    
    def setUp(self):
        self.config = OptimizationConfig()
    
    def test_default_values(self):
        """测试默认值"""
        self.assertEqual(self.config.n_calls, 20)
        self.assertEqual(self.config.n_initial_points, 5)
        self.assertEqual(self.config.optimizer, OptimizerType.BAYESIAN)
        self.assertTrue(self.config.early_stopping)
    
    def test_valid_config(self):
        """测试有效配置"""
        try:
            self.config.validate()
        except Exception as e:
            self.fail(f"Valid config should not raise exception: {e}")
    
    def test_invalid_n_calls(self):
        """测试无效的调用次数"""
        self.config.n_calls = 0
        with self.assertRaises(ConfigurationError):
            self.config.validate()
    
    def test_invalid_n_initial_points(self):
        """测试无效的初始点数"""
        self.config.n_initial_points = 0
        with self.assertRaises(ConfigurationError):
            self.config.validate()
    
    def test_n_initial_points_greater_than_n_calls(self):
        """测试初始点数大于调用次数"""
        self.config.n_calls = 10
        self.config.n_initial_points = 15
        with self.assertRaises(ConfigurationError):
            self.config.validate()
    
    @patch('src.utils.dependency_manager.is_available')
    def test_get_available_optimizers(self, mock_is_available):
        """测试获取可用优化器"""
        # 模拟scikit-optimize不可用
        mock_is_available.return_value = False
        available = self.config.get_available_optimizers()
        self.assertIn('random', available)
        self.assertIn('genetic', available)
        self.assertNotIn('bayesian', available)
        
        # 模拟scikit-optimize可用
        mock_is_available.return_value = True
        available = self.config.get_available_optimizers()
        self.assertIn('bayesian', available)
        self.assertIn('forest', available)


class TestAnsaConfig(unittest.TestCase):
    """ANSA配置测试"""
    
    def setUp(self):
        self.config = AnsaConfig()
    
    def test_default_values(self):
        """测试默认值"""
        self.assertEqual(self.config.ansa_executable, 'ansa')
        self.assertEqual(self.config.execution_timeout, 300)
        self.assertTrue(self.config.quality_check_enabled)
    
    def test_valid_config(self):
        """测试有效配置"""
        try:
            self.config.validate()
        except Exception as e:
            self.fail(f"Valid config should not raise exception: {e}")
    
    def test_invalid_element_length(self):
        """测试无效的单元长度"""
        self.config.min_element_length = 0
        with self.assertRaises(ConfigurationError):
            self.config.validate()
        
        self.config.min_element_length = 5.0
        self.config.max_element_length = 3.0
        with self.assertRaises(ConfigurationError):
            self.config.validate()
    
    def test_output_dir_creation(self):
        """测试输出目录创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_output'
            config = AnsaConfig(output_dir=output_path)
            self.assertTrue(output_path.exists())


class TestUnifiedParameterSpace(unittest.TestCase):
    """统一参数空间测试"""
    
    def setUp(self):
        self.param_space = UnifiedParameterSpace()
    
    def test_parameter_definitions(self):
        """测试参数定义"""
        param_names = self.param_space.get_parameter_names()
        self.assertIn('element_size', param_names)
        self.assertIn('perimeter_length', param_names)
        self.assertIn('quality_threshold', param_names)
    
    def test_get_parameter(self):
        """测试获取参数"""
        param = self.param_space.get_parameter('element_size')
        self.assertIsNotNone(param)
        if param is not None:
            self.assertEqual(param.name, 'element_size')
            self.assertEqual(param.param_type, ParameterType.FLOAT)
    
    def test_get_bounds(self):
        """测试获取边界"""
        bounds = self.param_space.get_bounds()
        self.assertIsInstance(bounds, list)
        self.assertTrue(len(bounds) > 0)
    
    def test_get_ansa_mapping(self):
        """测试获取ANSA映射"""
        mapping = self.param_space.get_ansa_mapping()
        self.assertIsInstance(mapping, dict)
        self.assertIn('element_size', mapping)
    
    def test_validate_bounds(self):
        """测试边界验证"""
        try:
            self.param_space.validate_bounds()
        except Exception as e:
            self.fail(f"Valid bounds should not raise exception: {e}")
    
    def test_validate_parameter_values(self):
        """测试参数值验证"""
        valid_values = {
            'element_size': 1.0,
            'quality_threshold': 0.5
        }
        try:
            self.param_space.validate_parameter_values(valid_values)
        except Exception as e:
            self.fail(f"Valid values should not raise exception: {e}")
        
        invalid_values = {
            'element_size': -1.0,  # 负值
            'unknown_param': 1.0   # 未知参数
        }
        with self.assertRaises(ValidationError):
            self.param_space.validate_parameter_values(invalid_values)
    
    @patch('src.utils.dependency_manager.is_available')
    def test_to_skopt_space_unavailable(self, mock_is_available):
        """测试scikit-optimize不可用时的空间转换"""
        mock_is_available.return_value = False
        with self.assertRaises(ConfigurationError):
            self.param_space.to_skopt_space()
    
    @patch('src.utils.dependency_manager.is_available')
    @patch('skopt.space.Real')
    @patch('skopt.space.Integer')
    def test_to_skopt_space_available(self, mock_integer, mock_real, mock_is_available):
        """测试scikit-optimize可用时的空间转换"""
        mock_is_available.return_value = True
        mock_real.return_value = MagicMock()
        mock_integer.return_value = MagicMock()
        
        try:
            space = self.param_space.to_skopt_space()
            self.assertIsInstance(space, list)
        except ImportError:
            # 如果实际环境中没有skopt，跳过测试
            self.skipTest("scikit-optimize not available in test environment")


class TestUnifiedConfigManager(unittest.TestCase):
    """统一配置管理器测试"""
    
    def setUp(self):
        self.config_manager = UnifiedConfigManager()
    
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.config_manager.optimization_config, OptimizationConfig)
        self.assertIsInstance(self.config_manager.ansa_config, AnsaConfig)
        self.assertIsInstance(self.config_manager.parameter_space, UnifiedParameterSpace)
    
    def test_validate_all_configs(self):
        """测试所有配置验证"""
        try:
            self.config_manager.validate_all_configs()
        except Exception as e:
            self.fail(f"Valid configs should not raise exception: {e}")
    
    def test_save_and_load_config(self):
        """测试配置保存和加载"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            # 修改一些配置
            self.config_manager.optimization_config.n_calls = 50
            self.config_manager.ansa_config.execution_timeout = 600
            
            # 保存配置
            self.config_manager.save_config(config_file)
            self.assertTrue(Path(config_file).exists())
            
            # 创建新的管理器并加载配置
            new_manager = UnifiedConfigManager(config_file)
            self.assertEqual(new_manager.optimization_config.n_calls, 50)
            self.assertEqual(new_manager.ansa_config.execution_timeout, 600)
            
        finally:
            # 清理临时文件
            if Path(config_file).exists():
                Path(config_file).unlink()
    
    def test_get_config_summary(self):
        """测试配置摘要"""
        summary = self.config_manager.get_config_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('optimization', summary)
        self.assertIn('parameter_space', summary)
        self.assertIn('ansa', summary)
    
    def test_load_invalid_config_file(self):
        """测试加载无效配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            config_file = f.name
        
        try:
            with self.assertRaises(ConfigurationError):
                UnifiedConfigManager(config_file)
        finally:
            Path(config_file).unlink()


if __name__ == '__main__':
    unittest.main()