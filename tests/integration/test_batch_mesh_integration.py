#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批处理网格集成测试

作者: Chel
创建日期: 2025-07-07
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluators.batch_mesh_improved import (
    AnsaBatchMeshRunner, AnsaBatchConfig, batch_mesh_with_params
)

class TestBatchMeshIntegration(unittest.TestCase):
    """批处理网格集成测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AnsaBatchConfig()
        self.runner = AnsaBatchMeshRunner(
            script_dir=Path(self.temp_dir),
            config=self.config
        )
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_batch_mesh_workflow(self):
        """测试完整的批处理网格工作流"""
        # 设置测试参数
        params = {
            'mesh_density': 4.0,
            'quality_threshold': 0.6,
            'distortion_distance': 20
        }
        
        # 运行批处理
        success = self.runner.run_batch_mesh(params)
        self.assertTrue(success)
        
        # 检查质量
        quality_results = self.runner.check_element_quality()
        self.assertIsNotNone(quality_results)
        self.assertIn('total_elements', quality_results)
        self.assertIn('bad_elements', quality_results)
        
        # 生成质量报告
        report_file = self.runner.generate_quality_report(quality_results)
        self.assertTrue(Path(report_file).exists())
        
        # 检查统计信息
        stats = self.runner.get_stats()
        self.assertIn('total_elements', stats)
        self.assertIn('execution_time', stats)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 无效参数测试
        invalid_params = {
            'mesh_density': -1.0,  # 无效的负值
            'quality_threshold': 2.0    # 无效的超出范围值
        }
        
        success = self.runner.run_batch_mesh(invalid_params)
        self.assertFalse(success)
        
        # 检查统计信息中的错误标记
        stats = self.runner.get_stats()
        self.assertFalse(stats['success'])
    
    def test_quality_thresholds(self):
        """测试质量阈值"""
        custom_thresholds = {
            'min_element_length': 1.0,
            'max_element_length': 5.0
        }
        
        # 使用自定义阈值进行质量检查
        quality_results = self.runner.check_element_quality(custom_thresholds)
        self.assertEqual(quality_results['thresholds']['min_length'], 1.0)
        self.assertEqual(quality_results['thresholds']['max_length'], 5.0)
    
    def test_config_validation(self):
        """测试配置验证"""
        # 设置无效配置
        self.config.min_element_length = -1.0
        self.config.max_element_length = 0.0
        
        # 验证配置
        is_valid, errors = self.config.validate()
        self.assertFalse(is_valid)
        self.assertTrue(len(errors) > 0)
    
    @patch('src.evaluators.batch_mesh_improved.ANSA_AVAILABLE', False)
    def test_mock_mode(self):
        """测试模拟模式"""
        # 在ANSA不可用时运行
        success = self.runner.run_batch_mesh()
        self.assertTrue(success)  # 模拟模式应该成功
        
        # 检查模拟的质量结果
        quality_results = self.runner.check_element_quality()
        self.assertIsNotNone(quality_results)
        self.assertIsInstance(quality_results['total_elements'], int)
    
    def test_batch_mesh_with_params_function(self):
        """测试批处理参数函数"""
        params = {
            'mesh_density': 4.0,
            'quality_threshold': 0.6
        }
        
        bad_elements = batch_mesh_with_params(params)
        self.assertIsInstance(bad_elements, int)
        self.assertLess(bad_elements, 100000)  # 应小于失败标记值99999

if __name__ == '__main__':
    unittest.main()