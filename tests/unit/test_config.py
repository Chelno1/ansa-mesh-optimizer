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

from src.config.config import (
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
        # 测试新增的 rule_fillet_width 参数
        self.assertIn('rule_fillet_width_1', param_names)
        self.assertIn('rule_fillet_width_2', param_names)
        self.assertIn('rule_fillet_width_3', param_names)
        self.assertIn('rule_fillet_width_4', param_names)
    
    def test_get_parameter(self):
        """测试获取参数"""
        param = self.param_space.get_parameter('distortion_distance')
        self.assertIsNotNone(param)
        if param is not None:
            self.assertEqual(param.name, 'distortion_distance')
            self.assertEqual(param.param_type, ParameterType.INTEGER)
    
    def test_get_bounds(self):
        """测试获取边界"""
        bounds = self.param_space.get_bounds()
        self.assertIsInstance(bounds, list)
        self.assertTrue(len(bounds) > 0)
    
    def test_get_ansa_mapping(self):
        """测试获取ANSA映射"""
        mapping = self.param_space.get_ansa_mapping()
        self.assertIsInstance(mapping, dict)
        self.assertIn('distortion_distance', mapping)
    
    def test_validate_bounds(self):
        """测试边界验证"""
        try:
            self.param_space.validate_bounds()
        except Exception as e:
            self.fail(f"Valid bounds should not raise exception: {e}")
    
    def test_validate_parameter_values(self):
        """测试参数值验证"""
        valid_values = {
            'distortion_distance': 20
        }
        try:
            self.param_space.validate_parameter_values(valid_values)
        except Exception as e:
            self.fail(f"Valid values should not raise exception: {e}")
        
        invalid_values = {
            'distortion_distance': -1.0,  # 负值
            'unknown_param': 1.0   # 未知参数
        }
        with self.assertRaises(ValidationError):
            self.param_space.validate_parameter_values(invalid_values)
    
    def test_rule_fillet_width_parameters(self):
        """测试 rule_fillet_width 参数"""
        param_names = self.param_space.get_parameter_names()
        
        # 确保所有 rule_fillet_width 参数都存在
        for i in range(1, 5):
            param_name = f'rule_fillet_width_{i}'
            self.assertIn(param_name, param_names)
            
            # 检查参数定义
            param = self.param_space.get_parameter(param_name)
            self.assertIsNotNone(param)
            if param is not None:
                self.assertEqual(param.param_type, ParameterType.FLOAT)
                # 检查每个参数的具体边界
                if i == 1:
                    self.assertEqual(param.bounds, (1.0, 5.0))
                elif i == 2:
                    self.assertEqual(param.bounds, (5.0, 12.0))
                elif i == 3:
                    self.assertEqual(param.bounds, (12.0, 25.0))
                elif i == 4:
                    self.assertEqual(param.bounds, (25.0, 40.0))
    
    def test_rule_fillet_width_ordering_constraint(self):
        """测试 rule_fillet_width 参数的递增约束"""
        # 测试有效的递增序列（使用实际边界内的值）
        valid_values = {
            'rule_fillet_width_1': 3.0,   # 在 (1.0, 5.0) 范围内
            'rule_fillet_width_2': 8.0,   # 在 (5.0, 12.0) 范围内
            'rule_fillet_width_3': 18.0,  # 在 (12.0, 25.0) 范围内
            'rule_fillet_width_4': 30.0   # 在 (25.0, 40.0) 范围内
        }
        try:
            self.param_space.validate_parameter_values(valid_values)
        except Exception as e:
            self.fail(f"Valid ascending values should not raise exception: {e}")
        
        # 测试无效的非递增序列
        invalid_values = {
            'rule_fillet_width_1': 4.0,   # 在 (1.0, 5.0) 范围内
            'rule_fillet_width_2': 6.0,   # 在 (5.0, 12.0) 范围内
            'rule_fillet_width_3': 15.0,  # 在 (12.0, 25.0) 范围内
            'rule_fillet_width_4': 28.0   # 在 (25.0, 40.0) 范围内，但违反递增约束
        }
        # 这个测试应该通过，因为值是递增的。让我们测试真正违反约束的情况
        invalid_values_2 = {
            'rule_fillet_width_1': 4.0,   # 在 (1.0, 5.0) 范围内
            'rule_fillet_width_2': 10.0,  # 在 (5.0, 12.0) 范围内
            'rule_fillet_width_3': 8.0,   # 违反递增约束：应该 > 10.0
            'rule_fillet_width_4': 30.0   # 在 (25.0, 40.0) 范围内
        }
        with self.assertRaises(ValidationError):
            self.param_space.validate_parameter_values(invalid_values_2)
        
        # 测试相等值（应该被拒绝）
        equal_values = {
            'rule_fillet_width_1': 3.0,   # 在 (1.0, 5.0) 范围内
            'rule_fillet_width_2': 8.0,   # 在 (5.0, 12.0) 范围内
            'rule_fillet_width_3': 8.0,   # 相等值违反严格递增约束
            'rule_fillet_width_4': 30.0   # 在 (25.0, 40.0) 范围内
        }
        with self.assertRaises(ValidationError):
            self.param_space.validate_parameter_values(equal_values)
    
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